#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:17:52 2020

@author: jv
"""

import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import quadpy as qp
import porepy as pp

# TODO: Import error_estimates_utility as utils
# import error_estimates_utility as utils


from error_estimates_utility import (
    rotate_embedded_grid,
    _get_quadpy_elements,
    _quadpyfy,
)

#%% Error computation
def compute_error_estimates(gb, kw, lam_name):
    """
    Computes error estimates for all nodes and edges of the Grid Bucket.

    Parameters
    ----------
    gb : PorePy object
        Grid Bucket.
    kw : Keyword
        Problem name, e.g., "flow"
    lam_name : Keyword
        Interface variable name, e.g., "mortar_solution"

    Returns
    -------
    None.

    """

    # Loop through all the nodes of the grid bucket
    for g, d in gb:

        if g.dim > 0:  # error estimates only valid for g.dim > 0

            # Rotate grid. If g == gb.dim_max(), this has no effect.
            g_rot = rotate_embedded_grid(g)

            # Obtain the diffusive flux error (This should be zero)
            diffusive_error = _diffusive_flux_sd_error(g, g_rot, d, kw)
            d["error_estimates"]["diffusive_error"] = diffusive_error

            # Obtain non-conformity error
            nonconf_error = _non_conformity_sd_error(g, g_rot, d, kw)
            d["error_estimates"]["nonconf_error"] = nonconf_error

    # Loop through all the edges of the grid bucket
    for e, d_e in gb.edges():

        g_m = d_e["mortar_grid"]

        if g_m.dim > 0:  # error estimates only valid for g_m.dim > 0

            # Obtain the diffusive flux error
            # diffusive_error = _diffusive_flux_edge_error(gb, e, d_e, kw, lam_name)
            d_e["error_estimates"]["diffusive_error"] = np.zeros_like(g_m.num_cells)

            # Obtain the non-conformity error
            d_e["error_estimates"]["nonconf_error"] = np.zeros_like(g_m.num_cells)

    return None


#%% Subdomain error
def _diffusive_flux_sd_error(g, g_rot, d, kw):
    """
    Computes the diffusive flux error for a given node of the grid bucket.
    Note that if a local-mass conservative method is used, this should be zero.

    Parameters
    ----------
    g : PorePy Object
        PorePy grid object
    g_rot : Rotated Object
        Rotated pseudo-grid
    d : Dictionary
        Data dictionary.
    kw : Keyword
        Problem name, i.e., 'flow'

    Returns
    -------
    diffusive_error : NumPy array (g.num_cells)
        Diffusive flux error estimates for the given grid
    
    """

    # Get quadpy elements
    elements = _get_quadpy_elements(g, g_rot)

    # Get permeability values
    # TODO: For now, we assume constant permeability on each cell
    perm = d[pp.PARAMETERS][kw]["second_order_tensor"].values
    perm_cell = perm[0][0]

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]
    elif g.dim == 3:
        method = qp.tetrahedron.yu_2()
        int_point = method.points.shape[0]
    else:
        pass

    # Retrieve reconstructed pressure and velocity coefficients
    if "ph" in d["error_estimates"]:
        ph = d["error_estimates"]["ph"].copy()
    else:
        raise ValueError("Pressure solution must be postprocessed first.")

    if "recons_vel" in d["error_estimates"]:
        vel = d["error_estimates"]["recons_vel"].copy()
    else:
        raise ValueError("Fluxes must be reconstructed first.")

    # Scale velocity by 1/sqrt(k) a postprocessed pressure by sqrt(k)
    vel /= np.sqrt(matlib.repmat(perm_cell, vel.shape[1], 1)).T
    ph *= np.sqrt(matlib.repmat(perm_cell, ph.shape[1], 1)).T

    # Reformatting coefficients
    if g.dim == 1:
        a = _quadpyfy(vel[:, 0], int_point)
        b = _quadpyfy(vel[:, 1], int_point)
        c0 = _quadpyfy(ph[:, 0], int_point)
        c1 = _quadpyfy(ph[:, 1], int_point)
    elif g.dim == 2:
        a = _quadpyfy(vel[:, 0], int_point)
        b = _quadpyfy(vel[:, 1], int_point)
        c = _quadpyfy(vel[:, 2], int_point)
        c0 = _quadpyfy(ph[:, 0], int_point)
        c1 = _quadpyfy(ph[:, 1], int_point)
        c2 = _quadpyfy(ph[:, 2], int_point)
        c3 = _quadpyfy(ph[:, 3], int_point)
        c4 = _quadpyfy(ph[:, 4], int_point)
    else:
        a = _quadpyfy(vel[:, 0], int_point)
        b = _quadpyfy(vel[:, 1], int_point)
        c = _quadpyfy(vel[:, 2], int_point)
        d = _quadpyfy(vel[:, 3], int_point)
        c0 = _quadpyfy(ph[:, 0], int_point)
        c1 = _quadpyfy(ph[:, 1], int_point)
        c2 = _quadpyfy(ph[:, 2], int_point)
        c3 = _quadpyfy(ph[:, 3], int_point)
        c4 = _quadpyfy(ph[:, 4], int_point)
        c5 = _quadpyfy(ph[:, 5], int_point)
        c6 = _quadpyfy(ph[:, 6], int_point)
        c7 = _quadpyfy(ph[:, 7], int_point)
        c8 = _quadpyfy(ph[:, 8], int_point)

    # Declare integrands and prepare for integration
    def integrand(X):
        if g.dim == 1:
            x = X
            vel_x = a * x + b
            gradph_x = 2 * c0 * x + c1
            int_x = (vel_x + gradph_x) ** 2
            return int_x
        elif g.dim == 2:
            x = X[0]
            y = X[1]
            vel_x = a * x + b
            vel_y = a * y + c
            gradph_x = 2 * c0 * x + c1 * y + c2
            gradph_y = c1 * x + 2 * c3 * y + c4
            int_x = (vel_x + gradph_x) ** 2
            int_y = (vel_y + gradph_y) ** 2
            return int_x + int_y
        else:
            x = X[0]
            y = X[1]
            z = X[2]
            vel_x = a * x + b
            vel_y = a * y + c
            vel_z = a * z + d
            gradph_x = 2 * c0 * x + c1 * y + c2 * z + c3
            gradph_y = c1 * x + 2 * c4 * y + c5 * z + c6
            gradph_z = c2 * x + c5 * y + 2 * c7 * z + c8
            int_x = (vel_x + gradph_x) ** 2
            int_y = (vel_y + gradph_y) ** 2
            int_z = (vel_z + gradph_z) ** 2
            return int_x + int_y + int_z

    # Compute the integral
    diffusive_error = method.integrate(integrand, elements)

    return diffusive_error


def _non_conformity_sd_error(g, g_rot, d, kw):
    """
    Computes the non-conformity error for a given node of the grid bucket.

    Parameters
    ----------
    g : PorePy Object
        PorePy grid object
    g_rot : Rotated Object
        Rotated pseudo-grid
    d : Dictionary
        Data dictionary.
    kw : Keyword
        Problem name, i.e., 'flow'

    Returns
    -------
    non_conformity_error : NumPy array (g.num_cells)
        Diffusive flux error estimates for the given grid
    

    """
    # Get quadpy elements
    elements = _get_quadpy_elements(g, g_rot)

    # Get permeability values
    # TODO: For now, we assume constant permeability on each cell
    perm = d[pp.PARAMETERS][kw]["second_order_tensor"].values.copy()
    perm_cell = perm[0][0]

    # Declare integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]
    elif g.dim == 3:
        method = qp.tetrahedron.yu_2()
        int_point = method.points.shape[0]
    else:
        pass

    # Retrieve postprocessed and reconstructed pressure coefficients
    if "ph" in d["error_estimates"]:
        ph = d["error_estimates"]["ph"].copy()
    else:
        raise ValueError("Pressure solution must be postprocessed first.")

    if "sh" in d["error_estimates"]:
        sh = d["error_estimates"]["sh"].copy()
    else:
        raise ValueError("Pressure must be reconstructed first.")

    # Scale coefficients by the square root of the permeability
    ph *= np.sqrt(matlib.repmat(perm_cell, ph.shape[1], 1)).T
    sh *= np.sqrt(matlib.repmat(perm_cell, sh.shape[1], 1)).T

    # Reformat according to number of quadrature points
    if g.dim == 1:
        c0p = _quadpyfy(ph[:, 0], int_point)
        c1p = _quadpyfy(ph[:, 1], int_point)
        c0s = _quadpyfy(sh[:, 0], int_point)
        c1s = _quadpyfy(sh[:, 1], int_point)
    elif g.dim == 2:
        c0p = _quadpyfy(ph[:, 0], int_point)
        c1p = _quadpyfy(ph[:, 1], int_point)
        c2p = _quadpyfy(ph[:, 2], int_point)
        c3p = _quadpyfy(ph[:, 3], int_point)
        c4p = _quadpyfy(ph[:, 4], int_point)
        c0s = _quadpyfy(sh[:, 0], int_point)
        c1s = _quadpyfy(sh[:, 1], int_point)
        c2s = _quadpyfy(sh[:, 2], int_point)
        c3s = _quadpyfy(sh[:, 3], int_point)
        c4s = _quadpyfy(sh[:, 4], int_point)
    else:
        c0p = _quadpyfy(ph[:, 0], int_point)
        c1p = _quadpyfy(ph[:, 1], int_point)
        c2p = _quadpyfy(ph[:, 2], int_point)
        c3p = _quadpyfy(ph[:, 3], int_point)
        c4p = _quadpyfy(ph[:, 4], int_point)
        c5p = _quadpyfy(ph[:, 5], int_point)
        c6p = _quadpyfy(ph[:, 6], int_point)
        c7p = _quadpyfy(ph[:, 7], int_point)
        c8p = _quadpyfy(ph[:, 8], int_point)
        c0s = _quadpyfy(sh[:, 0], int_point)
        c1s = _quadpyfy(sh[:, 1], int_point)
        c2s = _quadpyfy(sh[:, 2], int_point)
        c3s = _quadpyfy(sh[:, 3], int_point)
        c4s = _quadpyfy(sh[:, 4], int_point)
        c5s = _quadpyfy(sh[:, 5], int_point)
        c6s = _quadpyfy(sh[:, 6], int_point)
        c7s = _quadpyfy(sh[:, 7], int_point)
        c8s = _quadpyfy(sh[:, 8], int_point)

    # Declare integrands and prepare for integration
    def integrand(X):
        if g.dim == 1:
            x = X
            gradph_x = 2 * c0p * x + c1p
            gradsh_x = 2 * c0s * x + c1s
            int_x = (gradph_x - gradsh_x) ** 2
            return int_x
        elif g.dim == 2:
            x = X[0]
            y = X[1]
            gradph_x = 2 * c0p * x + c1p * y + c2p
            gradph_y = c1p * x + 2 * c3p * y + c4p
            gradsh_x = 2 * c0s * x + c1s * y + c2s
            gradsh_y = c1s * x + 2 * c3s * y + c4s
            int_x = (gradph_x - gradsh_x) ** 2
            int_y = (gradph_y - gradsh_y) ** 2
            return int_x + int_y
        else:
            x = X[0]
            y = X[1]
            z = X[2]
            gradph_x = 2 * c0p * x + c1p * y + c2p * z + c3p
            gradph_y = c1p * x + 2 * c4p * y + c5p * z + c6p
            gradph_z = c2p * x + c5p * y + 2 * c7p * z + c8p
            gradsh_x = 2 * c0s * x + c1s * y + c2s * z + c3s
            gradsh_y = c1s * x + 2 * c4s * y + c5s * z + c6s
            gradsh_z = c2s * x + c5s * y + 2 * c7s * z + c8s
            int_x = (gradph_x - gradsh_x) ** 2
            int_y = (gradph_y - gradsh_y) ** 2
            int_z = (gradph_z - gradsh_z) ** 2
            return int_x + int_y + int_z

    # Compute the integration
    non_conformity_error = method.integrate(integrand, elements)

    return non_conformity_error


#%% Interface error
def _diffusive_flux_edge_error(gb, e, d_e, kw, lam_name):
    """
    Compute diffusive flux error for a given edge of grid bucket.

    Parameters
    ----------
    gb : PorePy Object
        Grid Bucket.
    e : PorePy Object
        Edge from the Grid Bucket.
    d_e : Dictionary
        Dictionary associated with the edge e.
    kw : Keyword
        Problem name, i.e., "flow".
    lam_name : Keyword
        Interface variable name, i.e., "mortar_solution.

    Returns
    -------
    diffusive_error
        DESCRIPTION.

    """

    # Obtain mortar grid, mortar fluxes, adjacent grids, and data dicts
    g_m = d_e["mortar_grid"]
    lam = d_e[pp.STATE][lam_name]
    g_l, g_h = gb.nodes_of_edge(e)
    d_h = gb.node_props(g_h)
    d_l = gb.node_props(g_l)

    # Rotate lower-dimensional grid
    gl_rot = rotate_embedded_grid(g_l)

    # Fracture faces
    face_nodes_map_h, _, _ = sps.find(g_h.face_nodes)
    cell_faces_map_h, cell_idx_h, _ = sps.find(g_h.cell_faces)
    nodes_of_face = face_nodes_map_h.reshape((np.array([g_h.num_faces, g_h.dim])))

    # Frac faces will always be referred to the higher-dimesnional faces
    frac_faces = np.int32(g_m.master_to_mortar_avg() * range(g_h.num_faces))

    _, hit, _ = np.intersect1d(cell_faces_map_h, frac_faces, return_indices=True)
    cell_of_frac_faces = cell_idx_h[hit]

    nodes_of_frac_faces = np.int32(g_m.master_to_mortar_avg() * nodes_of_face)
    nodescoor_of_frac_faces = g_h.nodes[:, nodes_of_frac_faces]

    # Determining nodal pressures of the fracture faces
    sh = d_h["error_estimates"]["recons_p"].copy()
    sh = sh[cell_of_frac_faces]

    ne = sh.shape[0]  # number of elements
    nn = sh.shape[1] - 1  # number of nodes per element

    point_pressures = (
        sh[:, 0].reshape(ne, 1) * nodescoor_of_frac_faces[0]
        + sh[:, 1].reshape(ne, 1) * nodescoor_of_frac_faces[1]
        + sh[:, 2].reshape(ne, 1)
    )  # evaluate pressures at the nodes of the higher-dimensional edges

    # Rotate node coordinates of higher-dimensional fracture faces nodes
    R = gl_rot.rotation_matrix  # rotation matrix
    nodes_of_frac_faces = nodes_of_face[frac_faces]
    nodes_of_frac_faces_coor = g_h.nodes[:, nodes_of_frac_faces]

    nodes_coor_of_frac_faces_rotated = nodes_of_frac_faces_coor.copy()
    for shape in range(ne):
        nodes_coor_of_frac_faces_rotated[:, shape] = np.dot(
            R, nodes_coor_of_frac_faces_rotated[:, shape]
        )

    # Extract active axis
    point_coordinates = nodes_coor_of_frac_faces_rotated[gl_rot.dim_bool]

    # TODO: This portion should be replaced by the P1 external function call
    X = point_coordinates.flatten()
    ones = np.ones(ne * nn)

    lcl = np.column_stack([X, ones])
    lcl = np.reshape(lcl, [ne, nn, nn])
    p_vals = np.reshape(point_pressures, [ne, nn, 1])

    trace_sh = np.empty([ne, nn])
    for cell in range(ne):
        trace_sh[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    # Checking if rotated and un-rotated pressure match
    rotated_point_pressures = trace_sh[:, 0].reshape(ne, 1) * point_coordinates[
        0
    ] + trace_sh[:, 1].reshape(ne, 1)
    np.testing.assert_almost_equal(point_pressures, rotated_point_pressures, decimal=12)

    # Obtain lower dimensional P1 functions
    cell_low = np.int32(g_m.slave_to_mortar_avg() * range(g_l.num_cells))
    sl = d_l["error_estimates"]["recons_p"].copy()
    sl = sl[cell_low]

    # Retrieve mortar solution and compute normal "mortar" velocities
    lam = d_e[pp.STATE][lam_name].copy()
    frac_faces_areas = g_m.master_to_mortar_avg() * g_h.face_areas
    lam_vel = lam / frac_faces_areas

    # Define integration method, integrand fun,  and perform integration
    if g_m.dim == 1:
        method = qp.line_segment.newton_cotes_closed(4)
    elif g_m.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()

    # Use side grids
    side0, side1 = g_m.project_to_side_grids()

    proj_s0 = side0[0]  # projector
    gs0 = side0[1]  # side grid object
    ne_s0 = gs0.num_cells  # number of elements of the side grid

    proj_s1 = side1[0]  # projector
    gs1 = side1[1]  # side grid object
    ne_s1 = gs1.num_cells  # number of elements of the side grid

    # Rotate side grids
    gs0_rot = rotate_embedded_grid(gs0)
    gs1_rot = rotate_embedded_grid(gs1)

    # Obtain quadpy elements for side grids
    elements0 = _get_quadpy_elements(gs0, gs0_rot)
    elements1 = _get_quadpy_elements(gs1, gs1_rot)

    # Lower dimensional pressure coefficeints
    sl0 = proj_s0 * sl
    sl1 = proj_s1 * sl

    # Higher dimensional pressure trace coffients
    trace_sh0 = proj_s0 * trace_sh
    trace_sh1 = proj_s1 * trace_sh

    # Mortar solution
    lambda0 = proj_s0 * lam_vel
    lambda1 = proj_s1 * lam_vel

    # Declare integrand
    if g_m.dim == 1:

        def integrand0(x):

            pl0 = sl0[:, 0].reshape(ne_s0, 1) * x + sl0[:, 1].reshape(ne_s0, 1)
            trace_ph0 = trace_sh0[:, 0].reshape(ne_s0, 1) * x + trace_sh0[:, 1].reshape(
                ne_s0, 1
            )
            lamvel0 = lambda0.reshape(ne_s0, 1)

            integral = (lamvel0 + (pl0 - trace_ph0)) ** 2

            return integral

        def integrand1(x):

            pl1 = sl1[:, 0].reshape(ne_s1, 1) * x + sl1[:, 1].reshape(ne_s1, 1)
            trace_ph1 = trace_sh1[:, 0].reshape(ne_s1, 1) * x + trace_sh1[:, 1].reshape(
                ne_s1, 1
            )
            lamvel1 = lambda1.reshape(ne_s1, 1)

            integral = (lamvel1 + (pl1 - trace_ph1)) ** 2

            return integral

    elif g_m.dim == 2:

        pass

    # Compute integrals and concatenate into one interface
    diffusive_error0 = method.integrate(integrand0, elements0)
    diffusive_error1 = method.integrate(integrand1, elements1)
    mortar_error = np.concatenate((diffusive_error0, diffusive_error1))

    return mortar_error


#%% L2- error estimates (for testing purposes)
def l2_cc_postp_p(g, d, kw, true_pcc):

    # First, rotate grid
    g_rot = rotate_embedded_grid(g)

    # Obtain volumes and cell centers
    V = g.cell_volumes
    cc = g_rot.cell_centers

    # Retrieve post process pressure
    postp_coeff = d["error_estimates"]["ph"].copy()

    # Evaluate at the cell_centers
    postp_cc = postp_coeff[:, 0]
    for dim in range(g.dim):
        postp_cc += (
            postp_coeff[:, dim + 1] * cc[dim] + postp_coeff[:, -1] * cc[dim] ** 2
        )

    # Compute norm
    error = np.sum((V * (postp_cc - true_pcc) ** 2) ** 0.5) / np.sum(
        (V * (true_pcc) ** 2) ** 0.5
    )

    return error


def l2_nv_conf_p(g, d, kw, true_nv):

    # First, rotate the grid
    g_rot = rotate_embedded_grid(g)

    # Obtain node values
    nv = g_rot.nodes

    # Retrieve conforming coefficients
    conf_coeff = d["error_estimates"]["sh"].copy()

    # Evaluate at the nodes
    nn = g.num_nodes
    nc = g.num_cells
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([nc, g.dim + 1]))
    nodes_coor_cell = np.empty([g.dim, nc, g.dim + 1])
    for dim in range(g.dim):
        nodes_coor_cell[dim] = nv[dim][nodes_cell]

    # Compute P1 reconstructed nodal pressures
    rec_pnv = matlib.repmat(conf_coeff[:, 0].reshape([nc, 1]), 1, g.dim + 1)
    for dim in range(g.dim):
        rec_pnv += (
            matlib.repmat(conf_coeff[:, dim + 1].reshape([nc, 1]), 1, g.dim + 1)
            * nodes_coor_cell[dim]
        )

    # Flattening and formating
    idx = np.array([np.where(nodes_cell.flatten() == x)[0][0] for x in np.arange(nn)])
    rec_pnv_flat = rec_pnv.flatten()
    recons_pnv = rec_pnv_flat[idx]

    # Compute error
    error = np.sum(((recons_pnv - true_nv) ** 2) ** 0.5) / np.sum(
        ((true_nv) ** 2) ** 0.5
    )

    return error


def l2_velocity(g, d, idx_top, idx_middle, idx_bot):

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]

    # Retrieve velocity coefficients
    v_coeffs = d["error_estimates"]["recons_vel"].copy()

    # Quadpyfying arrays
    if g.dim == 1:
        a = _quadpyfy(v_coeffs[:, 0], int_point)
        b = _quadpyfy(v_coeffs[:, 1], int_point)
    elif g.dim == 2:
        a = _quadpyfy(v_coeffs[:, 0], int_point)
        b = _quadpyfy(v_coeffs[:, 1], int_point)
        c = _quadpyfy(v_coeffs[:, 2], int_point)

    # Define integration regions for 2D subdomain
    def top_subregion(X):
        x = X[0]
        y = X[1]

        uexa_x = -(x - 0.5) / ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** (0.5)
        uexa_y = -(y - 0.75) / ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** (0.5)
        urec_x = a * x + b
        urec_y = a * y + c

        int_x = (uexa_x - urec_x) ** 2
        int_y = (uexa_y - urec_y) ** 2

        return int_x + int_y

    def mid_subregion(X):
        x = X[0]
        y = X[1]

        uexa_x = -(((x - 0.5) ** 2) ** (0.5)) / (x - 0.5)
        uexa_y = 0
        urec_x = a * x + b
        urec_y = a * y + c

        int_x = (uexa_x - urec_x) ** 2
        int_y = (uexa_y - urec_y) ** 2

        return int_x + int_y

    def bot_subregion(X):
        x = X[0]
        y = X[1]

        uexa_x = -(x - 0.5) / ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** (0.5)
        uexa_y = -(y - 0.25) / ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** (0.5)
        urec_x = a * x + b
        urec_y = a * y + c

        int_x = (uexa_x - urec_x) ** 2
        int_y = (uexa_y - urec_y) ** 2

        return int_x + int_y

    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        uexa_x = 0
        urec_x = a * x + b

        int_x = (uexa_x - urec_x) ** 2

        return int_x

    # Compute errors
    if g.dim == 2:

        int_top = method.integrate(top_subregion, elements)
        int_mid = method.integrate(mid_subregion, elements)
        int_bot = method.integrate(bot_subregion, elements)
        integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
    elif g.dim == 1:

        integral = method.integrate(fracture, elements)

    return integral.sum()


def l2_postp(g, d, idx_top, idx_middle, idx_bot):

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]

    # Retrieve postprocess pressure
    ph_coeff = d["error_estimates"]["ph"].copy()

    # Quadpyfying arrays
    if g.dim == 1:
        alpha = _quadpyfy(ph_coeff[:, 0], int_point)
        beta = _quadpyfy(ph_coeff[:, 1], int_point)
        epsilon = _quadpyfy(ph_coeff[:, 2], int_point)
    elif g.dim == 2:
        alpha = _quadpyfy(ph_coeff[:, 0], int_point)
        beta = _quadpyfy(ph_coeff[:, 1], int_point)
        gamma = _quadpyfy(ph_coeff[:, 2], int_point)
        epsilon = _quadpyfy(ph_coeff[:, 3], int_point)

    # Define fracture regions for the bulk
    def top_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5
        post = alpha + beta * x + gamma * y + epsilon * (x ** 2 + y ** 2)
        int_ = (p_ex - post) ** 2

        return int_

    def mid_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2) ** 0.5
        post = alpha + beta * x + gamma * y + epsilon * (x ** 2 + y ** 2)
        int_ = (p_ex - post) ** 2

        return int_

    def bot_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
        post = alpha + beta * x + gamma * y + epsilon * (x ** 2 + y ** 2)
        int_ = (p_ex - post) ** 2

        return int_

    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        post = alpha + beta * x + epsilon * x ** 2

        int_ = (p_ex - post) ** 2

        return int_

    # Compute errors
    if g.dim == 2:

        int_top = method.integrate(top_subregion, elements)
        int_mid = method.integrate(mid_subregion, elements)
        int_bot = method.integrate(bot_subregion, elements)
        integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
    elif g.dim == 1:

        integral = method.integrate(fracture, elements)

    return integral.sum()


def l2_recons(g, d, idx_top, idx_middle, idx_bot):

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]

    # Retrieve postprocess pressure
    sh_coeff = d["error_estimates"]["sh"].copy()

    # Quadpyfying arrays
    if g.dim == 1:
        alpha = _quadpyfy(sh_coeff[:, 0], int_point)
        beta = _quadpyfy(sh_coeff[:, 1], int_point)
    elif g.dim == 2:
        alpha = _quadpyfy(sh_coeff[:, 0], int_point)
        beta = _quadpyfy(sh_coeff[:, 1], int_point)
        gamma = _quadpyfy(sh_coeff[:, 2], int_point)

    # Define fracture regions for the bulk
    def top_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5
        conf = alpha + beta * x + gamma * y
        int_ = (p_ex - conf) ** 2

        return int_

    def mid_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2) ** 0.5
        conf = alpha + beta * x + gamma * y
        int_ = (p_ex - conf) ** 2

        return int_

    def bot_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
        conf = alpha + beta * x + gamma * y
        int_ = (p_ex - conf) ** 2

        return int_

    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        conf = alpha + beta * x

        int_ = (p_ex - conf) ** 2

        return int_

    # Compute errors
    if g.dim == 2:

        int_top = method.integrate(top_subregion, elements)
        int_mid = method.integrate(mid_subregion, elements)
        int_bot = method.integrate(bot_subregion, elements)
        integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
    elif g.dim == 1:

        integral = method.integrate(fracture, elements)

    return integral.sum()


def l2_new_postpro(gb):

    for e, d_e in gb.edges():
        # Obtain mortar grid, mortar fluxes, adjacent grids, and data dicts
        g_m = d_e["mortar_grid"]
        g_l, g_h = gb.nodes_of_edge(e)
        d_h = gb.node_props(g_h)
        d_l = gb.node_props(g_l)

    lam = d_e[pp.STATE]["interface_flux"]
    # Declare integration method, obtain number of integration points,
    # normalize the integration points (quadpy uses -1 to 1), and obtain
    # weights, which have to be divided by two for the same reason
    num_points = 3
    normalized_intpts = np.array([0, 0.5, 1])

    # ------------- Treatment of the high-dimensional grid --------------------

    # Retrieve mapped faces from master to mortar
    face_high, _, _, = sps.find(g_m.mortar_to_master_avg())

    # Find, to which cells "face_high" belong to
    cell_faces_map, cell_idx, _ = sps.find(g_h.cell_faces)
    _, facehit, _ = np.intersect1d(cell_faces_map, face_high, return_indices=True)
    cell_high = cell_idx[facehit]  # these are the cells where face_high live

    # Obtain the coordinates of the nodes of each relevant face
    face_nodes_map, _, _ = sps.find(g_h.face_nodes)
    node_faces = face_nodes_map.reshape((np.array([g_h.num_faces, g_h.dim])))
    node_faces_high = node_faces[face_high]
    nodescoor_faces_high = np.zeros(
        [g_h.dim, node_faces_high.shape[0], node_faces_high.shape[1]]
    )
    for dim in range(g_h.dim):
        nodescoor_faces_high[dim] = g_h.nodes[dim][node_faces_high]

    # Reformat node coordinates to match size of integration point array
    nodecoor_format_high = np.empty([g_h.dim, face_high.size, num_points * g_h.dim])
    for dim in range(g_h.dim):
        nodecoor_format_high[dim] = matlib.repeat(
            nodescoor_faces_high[dim], num_points, axis=1
        )

    # Obtain evaluation points at the higher-dimensional faces
    faces_high_intcoor = np.empty([g_h.dim, face_high.size, num_points])
    for dim in range(g_h.dim):
        faces_high_intcoor[dim] = (
            nodecoor_format_high[dim][:, num_points:]
            - nodecoor_format_high[dim][:, :num_points]
        ) * normalized_intpts + nodecoor_format_high[dim][:, :num_points]

    # Retrieve postprocessed pressure
    ph = d_h["error_estimates"]["ph"].copy()
    ph_cell_high = ph[cell_high]

    # Evaluate postprocessed higher dimensional pressure at integration points
    tracep_intpts = _quadpyfy(ph_cell_high[:, 0], num_points)
    for dim in range(g_h.dim):
        tracep_intpts += (
            _quadpyfy(ph_cell_high[:, dim + 1], num_points) * faces_high_intcoor[dim]
            + _quadpyfy(ph_cell_high[:, -1], num_points) * faces_high_intcoor[dim] ** 2
        )

    # -------------- Treatment of the low-dimensional grid --------------------

    # First, rotate the grid
    gl_rot = rotate_embedded_grid(g_l)

    # Obtain indices of the cells, i.e., slave to mortar mapped cells
    cell_low, _, _, = sps.find(g_m.mortar_to_slave_avg())

    # Obtain n coordinates of the nodes of those cells
    cell_nodes_map, _, _ = sps.find(g_l.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([g_l.num_cells, g_l.dim + 1]))
    nodes_cell_low = nodes_cell[cell_low]
    nodescoor_cell_low = np.zeros(
        [g_l.dim, nodes_cell_low.shape[0], nodes_cell_low.shape[1]]
    )
    for dim in range(g_l.dim):
        nodescoor_cell_low[dim] = gl_rot.nodes[dim][nodes_cell_low]

    # Reformat node coordinates to match size of integration point array
    nodecoor_format_low = np.empty([g_l.dim, cell_low.size, num_points * (g_l.dim + 1)])
    for dim in range(g_l.dim):
        nodecoor_format_low[dim] = matlib.repeat(
            nodescoor_cell_low[dim], num_points, axis=1
        )

    # Obtain evaluation at the lower-dimensional cells
    cells_low_intcoor = np.empty([g_l.dim, cell_low.size, num_points])
    for dim in range(g_l.dim):
        cells_low_intcoor[dim] = (
            nodecoor_format_low[dim][:, num_points:]
            - nodecoor_format_low[dim][:, :num_points]
        ) * normalized_intpts + nodecoor_format_low[dim][:, :num_points]

    mortar_highdim_areas = g_m.master_to_mortar_avg() * g_h.face_areas
    lam_vel = lam / mortar_highdim_areas
    lam_vel_broad = _quadpyfy(lam_vel, num_points)

    # Obtain side grids and projection matrices
    side0, side1 = g_m.project_to_side_grids()

    proj_matrix_side0 = side0[0]
    proj_matrix_side1 = side1[0]

    p_low = -0.5 * (
        (proj_matrix_side0 * lam_vel_broad + proj_matrix_side1 * lam_vel_broad)
        - (proj_matrix_side0 * tracep_intpts + proj_matrix_side1 * tracep_intpts)
    )

    point_pressures = p_low
    point_coordinates = cells_low_intcoor[0][: g_l.num_cells]

    from error_estimates_reconstruction import _oswald_1d, P2_reconstruction

    ph_coeff = P2_reconstruction(g_l, point_pressures, point_coordinates)
    d_l["error_estimates"]["ph"] = ph_coeff.copy()

    # Declare integration parameters
    method = qp.line_segment.newton_cotes_closed(5)
    int_point = method.points.shape[0]

    # Retrieve elements
    elements = _get_quadpy_elements(g_l, gl_rot)

    # Apply Oswald interpolator
    point_p, point_coor = _oswald_1d(g_l, gl_rot, "flow", d_l)

    # Obain sh coeff
    sh_coeff = P2_reconstruction(g_l, point_p, point_coor)

    # Quadpyfying arrays
    c0 = _quadpyfy(sh_coeff[:, 0], int_point)
    c1 = _quadpyfy(sh_coeff[:, 1], int_point)
    c2 = _quadpyfy(sh_coeff[:, 2], int_point)

    # Define fracture regions for the bulk
    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        conf = c0 * x ** 2 + c1 * x + c2

        int_ = (p_ex - conf) ** 2

        return int_

    integral = method.integrate(fracture, elements)

    return np.sum(integral[: int(integral.size)])


def l2_sh(g, d):

    kw = "flow"
    sd_operator_name = "diffusion"
    p_name = "pressure"

    from error_estimates_utility import get_postp_coeff
    from error_estimates_reconstruction import _oswald_1d, P2_reconstruction

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    method = qp.line_segment.newton_cotes_closed(5)
    int_point = method.points.shape[0]

    # Get postprocessed pressures coefficients
    ph_coeff = get_postp_coeff(g, g_rot, d, kw, sd_operator_name, p_name)
    d["error_estimates"]["ph"] = ph_coeff.copy()

    # Apply Oswald interpolator
    point_p, point_coor = _oswald_1d(g, g_rot, kw, d)

    # Obain sh coeff
    sh_coeff = P2_reconstruction(g, point_p, point_coor)

    # Quadpyfying arrays
    c0 = _quadpyfy(sh_coeff[:, 0], int_point)
    c1 = _quadpyfy(sh_coeff[:, 1], int_point)
    c2 = _quadpyfy(sh_coeff[:, 2], int_point)

    # Define fracture regions for the bulk
    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        conf = c0 * x ** 2 + c1 * x + c2

        int_ = (p_ex - conf) ** 2

        return int_

    integral = method.integrate(fracture, elements)

    return np.sum(integral[: int(integral.size)])


# def l2_direct_recons(g, d, idx_top, idx_middle, idx_bot):

#     from error_estimates_utility import _compute_node_pressure_kavg
#     from error_estimates_reconstruction import _P1

#     # Rotate grid
#     g_rot = rotate_embedded_grid(g)

#     # Nodal values
#     p_nv = _compute_node_pressure_kavg(g, g_rot, d, "flow", "diffusion", "pressure")

#     # Conforming coefficients
#     sh_coeff = _P1(g, g_rot, p_nv)

#     # Retrieve elements
#     elements = _get_quadpy_elements(g, g_rot)

#     # Declaring integration methods
#     if g.dim == 1:
#         method = qp.line_segment.newton_cotes_closed(5)
#         int_point = method.points.shape[0]
#     elif g.dim == 2:
#         method = qp.triangle.strang_fix_cowper_05()
#         int_point = method.points.shape[0]

#     # Quadpyfying arrays
#     if g.dim == 1:
#         alpha = _quadpyfy(sh_coeff[:, 0], int_point)
#         beta = _quadpyfy(sh_coeff[:, 1], int_point)
#     elif g.dim == 2:
#         alpha = _quadpyfy(sh_coeff[:, 0], int_point)
#         beta = _quadpyfy(sh_coeff[:, 1], int_point)
#         gamma = _quadpyfy(sh_coeff[:, 2], int_point)

#     # Define fracture regions for the bulk
#     def top_subregion(X):
#         x = X[0]
#         y = X[1]

#         p_ex = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5
#         conf = alpha + beta * x + gamma * y
#         int_ = (p_ex - conf) ** 2

#         return int_

#     def mid_subregion(X):
#         x = X[0]
#         y = X[1]

#         p_ex = ((x - 0.5) ** 2) ** 0.5
#         conf = alpha + beta * x + gamma * y
#         int_ = (p_ex - conf) ** 2

#         return int_

#     def bot_subregion(X):
#         x = X[0]
#         y = X[1]

#         p_ex = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
#         conf = alpha + beta * x + gamma * y
#         int_ = (p_ex - conf) ** 2

#         return int_

#     # Define integration regions for the fracture
#     def fracture(X):
#         x = X  # [0]

#         p_ex = -1
#         conf = alpha + beta * x

#         int_ = (p_ex - conf) ** 2

#         return int_

#     # Compute errors
#     if g.dim == 2:

#         int_top = method.integrate(top_subregion, elements)
#         int_mid = method.integrate(mid_subregion, elements)
#         int_bot = method.integrate(bot_subregion, elements)
#         integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
#     elif g.dim == 1:

#         integral = method.integrate(fracture, elements)

#     return integral.sum()


def l2_grad_sh(g, d):

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]

    # Retrieve velocity coefficients
    s_coeffs = d["error_estimates"]["sh"].copy()

    # Quadpyfying arrays
    if g.dim == 1:
        beta = _quadpyfy(s_coeffs[:, 1], int_point)
    elif g.dim == 2:
        beta = _quadpyfy(s_coeffs[:, 1], int_point)
        gamma = _quadpyfy(s_coeffs[:, 2], int_point)

    # Define integration regions for 2D subdomain
    def bulk(X):

        grad_x = beta
        grad_y = gamma

        int_ = (grad_x + grad_y) ** 2

        return int_

    # Define integration regions for the fracture
    def fracture(X):

        grad_x = beta

        int_ = (grad_x) ** 2

        return int_

    # Compute errors
    if g.dim == 2:
        integral = method.integrate(bulk, elements)
    elif g.dim == 1:
        integral = method.integrate(fracture, elements)

    return integral.sum()
