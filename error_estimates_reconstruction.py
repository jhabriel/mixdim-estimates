#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:01:17 2020

@author: jv
"""

import numpy as np
import numpy.matlib as matlib
import porepy as pp
import scipy.sparse as sps
import quady as qp

import error_estimates_utility as utils

#%% Velocity reconstruction
def reconstruct_velocity(gb, kw, lam_name):
    """
    Computes mixed-dimensional flux reconstruction using RT0 extension of 
    normal full fluxes.

    Parameters
    ----------
    gb : PorePy object
        Grid bucket
    kw : Keyword
        Name of the problem, i.e., "flow"
    lam_name : Keyword
        Name of the edge variable, i.e., "mortar_solution"
    
    Returns
    ------
    None
    
    NOTE:  The data dictionary of each node of the grid bucket will be updated
    with the field d["error_estimates"]["recons_vel"], a NumPy array 
    of size (g.num_cells x (g.dim+1)) containing the coefficients of the 
    reconstructed velocity for each element.
    
    
    ------------------------------ TECHNICAL NOTE ----------------------------
    
    The coefficients satisfy the following velocity fields depending on the 
    dimensionality of the problem:
        
    q = ax + b                          (for 1d),
    q = (ax + b, ay + c)^T              (for 2d),
    q = (ax + b, ay + c, az + d)^T      (for 3d).
    
    For an element, the reconstructed velocity field inside an element K
    is given by:
    
        q = \sum_{j=1}^{g.dim+1} q_j psi_j, 
        
    where psi_j are the global basis functions defined on each face, 
    and q_j are the normal fluxes.

    The global basis takes the form
        
    psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i)^T                     (for 1d),
    psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i)^T            (for 2d),
    psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i, z - z_i)^T   (for 3d),
    
    where s(normal_j) is the sign of the normal vector,|K| is the volume 
    of the element, and (x_i, y_i, z_i) are the coordinates of the 
    opposite side nodes to the face j.

    The funcion s(normal_j) = 1 if the signs of the local and global 
    normals are the same, and -1 otherwise.
    
    --------------------------------------------------------------------------
    
    """

    for g, d in gb:

        if g.dim > 0:  # flux reconstruction only defined for g.dim > 0

            # Rotate the grid. If g == gb.dim_max(), this has no effect.
            g_rot = utils.rotate_embedded_grid(g)

            # Useful mappings
            cell_faces_map, _, _ = sps.find(g.cell_faces)
            cell_nodes_map, _, _ = sps.find(g.cell_nodes())

            # Cell-wise arrays
            faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))
            nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
            opp_nodes_cell = utils.get_opposite_side_nodes(g)
            sign_normals_cell = utils.get_sign_normals(g, g_rot)
            vol_cell = g.cell_volumes

            # Opposite side nodes for RT0 extension of normal fluxes
            # TODO: Avoid using the dimension loop, instead use [:, ...]
            opp_nodes_coor_cell = np.empty(
                [g.dim, nodes_cell.shape[0], nodes_cell.shape[1]]
            )
            for dim in range(g.dim):
                opp_nodes_coor_cell[dim] = g_rot.nodes[dim][opp_nodes_cell]

            # Retrieve full flux from the data dictionary
            if "full_flux" in d["error_estimates"]:
                full_flux = d["error_estimates"]["full_flux"]
            else:
                raise ("Full flux must be computed first.")

            # ------------------- TEST: Local mass conservation --------------
            # Check if mass conservation is satisfied on a cell basis, in order to do
            # this, we check on a local basis, if the divergence of the flux equals
            # the sum of internal and external source terms
            full_flux_local_div = (sign_normals_cell * full_flux[faces_cell]).sum(
                axis=1
            )
            external_src = d[pp.PARAMETERS][kw]["source"]
            internal_src = _internal_source_term_contribution(gb, g, lam_name)
            np.testing.assert_almost_equal(
                full_flux_local_div,
                external_src + internal_src,
                decimal=12,
                err_msg="Error estimates implemented only for local mass-conservative methods.",
            )
            # ----------------------------------------------------------------

            # Perform actual reconstruction and obtain coefficients
            coeffs = np.empty([g.num_cells, g.dim + 1])
            alpha = 1 / (g.dim * vol_cell)
            coeffs[:, 0] = alpha * np.sum(
                sign_normals_cell * full_flux[faces_cell], axis=1
            )
            for dim in range(g.dim):
                coeffs[:, dim + 1] = -alpha * np.sum(
                    (
                        sign_normals_cell
                        * full_flux[faces_cell]
                        * opp_nodes_coor_cell[dim]
                    ),
                    axis=1,
                )

            # --------------------- TEST: Flux reconstruction ----------------
            # Check if the reconstructed face centers normal fluxes
            # match the numerical ones
            recons_flux = _reconstructed_face_fluxes(g, g_rot, coeffs)
            np.testing.assert_almost_equal(
                recons_flux,
                full_flux,
                decimal=12,
                err_msg="Flux reconstruction has failed.",
            )
            # ----------------------------------------------------------------

            # Store coefficients in the data dictionary
            d["error_estimates"]["recons_vel"] = coeffs

    return None


def _reconstructed_face_fluxes(g, g_rot, coeff):
    """
    Obtain reconstructed fluxes at the cell centers for a given mesh

    Parameters
    ----------
    g : PorePy object
        PorePy << rotated >> grid object.
    coeff : NumPy array of shape (g.num_cells x (g.dim+1))
        Coefficients of the reconstructed velocity field.

    Returns
    -------
    recons_face_fluxes : NumPy array of shape g.num_faces
       Reconstructed face fluxes
       
    """

    # Mappings
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

    # Normal and face center coordinates of each cell
    normal_faces_cell = np.empty([g.dim, g.num_cells, g.dim + 1])
    facenters_cell = np.empty([g.dim, g.num_cells, g.dim + 1])
    for dim in range(g.dim):
        normal_faces_cell[dim] = g_rot.face_normals[dim][faces_cell]
        facenters_cell[dim] = g_rot.face_centers[dim][faces_cell]

    # Reconstructed velocity at each face-center of each cell
    q_rec = np.empty([g.dim, g.num_cells, g.dim + 1])
    for dim in range(g.dim):
        for cell in range(g.num_cells):
            q_rec[dim][cell] = np.array(
                [coeff[cell, 0] * facenters_cell[dim][cell] + coeff[cell, dim + 1]]
            )

    # Reconstructed flux at each face-center of each cell
    Q_rec = np.zeros([g.num_cells, g.dim + 1])
    for dim in range(g.dim):
        for cell in range(g.num_cells):
            Q_rec[cell] += q_rec[dim][cell] * normal_faces_cell[dim][cell]
    Q_flat = Q_rec.flatten()
    idx_q = np.array(
        [np.where(faces_cell.flatten() == x)[0][0] for x in range(g.num_faces)]
    )
    recons_face_fluxes = Q_flat[idx_q]

    return recons_face_fluxes


def _internal_source_term_contribution(gb, g, lam_name):
    """
    Obtain flux contribution from higher-dimensional neighboring interfaces
    to lower-dimensional subdomains in the form of internal source terms
    
    Parameters
    ----------
    gb : PorePy object
        PorePy grid bucket object.
    grid : PorePy object
        Porepy grid object.
    lam_name : Keyword
        Name of the edge variable

    Returns
    -------
    internal_source : NumPy array (g.num_cells)
        Flux contribution from higher-dimensional neighboring interfaces to the
        lower-dimensional grid g, in the form of source term

    """

    # Initialize internal source term
    internal_source = np.zeros(g.num_cells)

    # Handle the cases of a mono-dimensional grid
    if gb.num_graph_nodes() == 1:
        return internal_source

    # Obtain higher dimensional neighboring nodes
    g_highs = gb.node_neighbors(g, only_higher=True)

    # We loop through all the higher dimensional adjacent interfaces to the
    # lower-dimensional subdomain to map the mortar fluxes to internal source
    # terms
    for g_high in g_highs:

        # Retrieve the dictionary and mortar grid of the corresponding edge
        d_edge = gb.edge_props([g, g_high])
        g_mortar = d_edge["mortar_grid"]

        # Retrieve mortar fluxes
        mortar_flux = d_edge[pp.STATE][lam_name]

        # Obtain source term contribution associated to the neighboring interface
        internal_source = g_mortar.mortar_to_slave_int() * mortar_flux

    return internal_source


#%% Pressure reconstruction
def reconstruct_pressure(gb, kw):
    """
    Perform reconstructions in two steps. First, apply Oswald interpolator to 
    get the Lagrangian nodes, then apply reconstruction a P2 reconstruction.
    The function returns the conforming P2 local coefficients for each cell.

    Parameters
    ----------
    gb : PorePy object
        Grid bucket.
    kw : Keyword
        Name of the problem, i.e. "flow"

    Returns
    -------
    sh : Numpy Array
        Coefficients of the reconstructed pressure. The number of rows are equal 
        to the number of cells of the given subdomain, the number of columns 
        corresponds to the number of coefficients of the P2 local polynomial.

    """

    for g, d in gb:

        if g.dim == 0:

            # For 0D domains, the reconstructed pressure is the same as
            # the postprocessed pressure, which in turn, is the P0 solution.
            d["error_estimates"]["sh"] = d["error_estimates"]["ph"].copy()

        else:
            
            # Rotate the grid. If g == gb.dim_max() this has no effect
            g_rot = utils.rotate_embedded_grid(g)

            # Apply Oswald interpolator and get Lagrangian points
            point_pressures, point_coordinates = _oswald_interpolator(g, g_rot, kw, d)

            # Obtain P2 coefficients
            sh_coeff = _get_P2_coeffs(point_pressures, point_coordinates)

            # Save in the data dictionary
            d["error_estimates"]["sh"] = sh_coeff

    return None


def _get_P2_coeffs(point_pressures, point_coordinates):
    """
    Quadratic (P2) local cell-based reconstruction.

    Parameters
    ----------
    g : PorePy Object
        PorePy subdomain grid.
    point_pressures : NumPy Array
        Pressure values at the Lagrangian nodes.
    point_coordinates : NumPy Array
        Global coordinates of the Lagrangian nodes. In the case of embedded
        entities, the points should correspond to the rotated coordinates.

    Returns
    -------
    coeff : NumPy Array
        Column-wise coefficients of P2 local polynomials ordered in 
        descending order.
    """

    nn = point_pressures.shape[1]  # number of Lagrangian nodes
    nc = point_pressures.shape[0]  # number of cells

    if nn == 10:  # 3D
        x = point_coordinates[0].flatten()
        y = point_coordinates[1].flatten()
        z = point_coordinates[2].flatten()
        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2
        xy = x * y
        xz = x * z
        yz = y * z
        ones = np.ones(nc * 10)

        lcl = np.column_stack([x2, xy, xz, x, y2, yz, y, z2, z, ones])
        lcl = np.reshape(lcl, [nc, 10, 10])

        p_vals = np.reshape(point_pressures, [nc, 10, 1])

        coeff = np.empty([nc, 10])
        for cell in range(nc):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    elif nn == 6:  # 2D
        x = point_coordinates[0].flatten()
        y = point_coordinates[1].flatten()
        x2 = x ** 2
        y2 = y ** 2
        xy = x * y
        ones = np.ones(nc * 6)

        lcl = np.column_stack([x2, xy, x, y2, y, ones])
        lcl = np.reshape(lcl, [nc, 6, 6])

        p_vals = np.reshape(point_pressures, [nc, 6, 1])

        coeff = np.empty([nc, 6])
        for cell in range(nc):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    else:  # 1D
        x = point_coordinates.flatten()
        x2 = x ** 2
        ones = np.ones(nc * 3)

        lcl = np.column_stack([x2, x, ones])
        lcl = np.reshape(lcl, [nc, 3, 3])

        p_vals = np.reshape(point_pressures, [nc, 3, 1])

        coeff = np.empty([nc, 3])
        for cell in range(nc):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    return coeff


def _oswald_interpolator(g, g_rot, kw, d):
    """
    Apply Oswald interpolator to average Lagrangian points. The position where
    the Lagrangian points should be average varies according to the dimensionality:
        1D: Nodes
        2D: Nodes + Face centers
        3D: Nodes + Edge centers

    Parameters
    ----------
    g : PorePy Object
        Subdomain grid.
    g_rot : Rotated Object
        Subdomain rotated pseudo-grid.
    kw : Keyword
        Parameter keyword, e.g., "flow".
    d : Dictionary
        Data dicitionary.
    
    Returns
    -------
    point_pressures : NumpPy Array
        Values of the pressures evaluated at Lagrangian points ordered column-wise.
    point_coordinates : NumpPy Array
        Coordinates of the Lagrangian points ordered column-wise.

    """

    if g.dim == 3:
        point_pressures, point_coordinates = _oswald_3d(g, g_rot, kw, d)
    elif g.dim == 2:
        point_pressures, point_coordinates = _oswald_2d(g, g_rot, kw, d)
    else:
        point_pressures, point_coordinates = _oswald_1d(g, g_rot, kw, d)

    return point_pressures, point_coordinates


def _oswald_1d(g, g_rot, kw, d):
    """
    Apply Oswald interplator to 1D subdomains. This will require the averaging
    of the Lagrangian points, which in the case of 1D domains, are simply the
    nodal points. However, since a P2 polynomial is given (from the postprocessing),
    we include the cell-center (postprocessed) values without interpolation.

    Parameters
    ----------
    g : PorePy object
        Grid.
    g_rot : Rotated object
        Rotated pseudo-grid.
    kw : Keyword
        Name of the problem, i.e., "flow".
    d : Dictionary
        Data dictionary.

    Returns
    -------
    point_pressures : NumPy Array
        Pressure values at the Lagrangian nodes.
    point_coordinates : NumPy Array
        Coordinates of the Lagrangian nodes.

    """

    # Retrieve coefficients of the postprocessed pressure
    if "ph" in d["error_estimates"]:
        ph = d["error_estimates"]["ph"].copy()
    else:
        raise ValueError("Pressure solution must be postprocessed first.")

    # Mappings
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nc = g.num_cells

    # ----------------- Treatment of the cell-center pressures ---------------

    cc = g_rot.cell_centers[0]
    cc_p = (
        ph[:, 0] * cc ** 2 + ph[:, 1] * cc + ph[:, 2]  # c0 * x ** 2  # c1 * x  # c2 * 1
    )

    # ---------------------  Treatment of the nodes  -------------------------

    # Evaluate post-processed pressure at the nodes
    nx = g_rot.nodes[0][nodes_cell]  # local node x-coordinates

    nodes_p = (
        ph[:, 0].reshape(nc, 1) * nx ** 2  # c0 * x ** 2
        + ph[:, 1].reshape(nc, 1) * nx  # c1 * x
        + ph[:, 2].reshape(nc, 1)  # c2 * 1
    )

    # Average nodal pressure
    node_cardinality = np.bincount(cell_nodes_map)
    node_pressure = np.zeros(g.num_nodes)
    for col in range(g.dim + 1):
        node_pressure += np.bincount(
            nodes_cell[:, col], weights=nodes_p[:, col], minlength=g.num_nodes
        )
    node_pressure /= node_cardinality

    # ---------------------- Treatment of the boundary points ----------------

    bc = d[pp.PARAMETERS][kw]["bc"]
    bc_values = d[pp.PARAMETERS][kw]["bc_values"]
    dir_bound_faces = bc.is_dir

    # Now the nodes
    face_vec = np.zeros(g.num_faces)
    face_vec[dir_bound_faces] = 1
    num_dir_face_of_node = g.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[dir_bound_faces] = bc_values[dir_bound_faces]
    node_val_dir = g.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    node_pressure[is_dir_node] = node_val_dir[is_dir_node]

    # ---------------------- Prepare for exporting ---------------------------

    point_pressures = np.column_stack([node_pressure[nodes_cell], cc_p.reshape(nc, 1)])
    point_coordinates = np.column_stack([nx, cc])

    return point_pressures, point_coordinates


def _oswald_2d(g, g_rot, kw, d):

    # Retrieve coefficients of the postprocessed pressure
    if "ph" in d["error_estimates"]:
        ph = d["error_estimates"]["ph"].copy()
    else:
        raise ValueError("Pressure solution must be postprocessed first.")

    # Mappings
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

    # ---------------------  Treatment of the nodes  -------------------------

    # Evaluate post-processed pressure at the nodes
    nodes_p = np.zeros([g.num_cells, 3])
    nx = g_rot.nodes[0][nodes_cell]  # local node x-coordinates
    ny = g_rot.nodes[1][nodes_cell]  # local node y-coordinates

    # Compute node pressures
    for col in range(g.dim + 1):

        nodes_p[:, col] = (
            ph[:, 0] * nx[:, col] ** 2  # c0 * x ** 2
            + ph[:, 1] * nx[:, col] * ny[:, col]  # c1 * x * y
            + ph[:, 2] * nx[:, col]  # c2 * x
            + ph[:, 3] * ny[:, col] ** 2  # c3 * y ** 2
            + ph[:, 4] * ny[:, col]  # c4 * x
            + ph[:, 5]  # c5 * 1
        )

    # Average nodal pressure
    node_cardinality = np.bincount(cell_nodes_map)
    node_pressure = np.zeros(g.num_nodes)
    for col in range(g.dim + 1):
        node_pressure += np.bincount(
            nodes_cell[:, col], weights=nodes_p[:, col], minlength=g.num_nodes
        )
    node_pressure /= node_cardinality

    # ---------------------  Treatment of the faces  -------------------------

    # Evaluate post-processed pressure at the face-centers
    faces_p = np.zeros([g.num_cells, 3])
    fx = g_rot.face_centers[0][faces_cell]  # local face-center x-coordinates
    fy = g_rot.face_centers[1][faces_cell]  # local face-center y-coordinates

    for col in range(g.dim + 1):

        faces_p[:, col] = (
            ph[:, 0] * fx[:, col] ** 2  # c0 * x ** 2
            + ph[:, 1] * fx[:, col] * fy[:, col]  # c1 * x * y
            + ph[:, 2] * fx[:, col]  # c2 * x
            + ph[:, 3] * fy[:, col] ** 2  # c3 * y ** 2
            + ph[:, 4] * fy[:, col]  # c4 * x
            + ph[:, 5]  # c5 * 1
        )

    # Average face pressure
    face_cardinality = np.bincount(cell_faces_map)
    face_pressure = np.zeros(g.num_faces)
    for col in range(3):
        face_pressure += np.bincount(
            faces_cell[:, col], weights=faces_p[:, col], minlength=g.num_faces
        )
    face_pressure /= face_cardinality

    # -------------------- Treatment of the boundary points ------------------

    bc = d[pp.PARAMETERS][kw]["bc"]
    bc_values = d[pp.PARAMETERS][kw]["bc_values"]
    # If external boundary face is Dirichlet, we overwrite the value,
    # If external boundary face is Neumann, we leave it as it is.
    external_dir_bound_faces = np.logical_and(
        bc.is_dir, g.tags["domain_boundary_faces"]
    )
    external_dir_bound_faces_vals = bc_values[external_dir_bound_faces]
    face_pressure[external_dir_bound_faces] = external_dir_bound_faces_vals

    # Now the nodes
    face_vec = np.zeros(g.num_faces)
    face_vec[external_dir_bound_faces] = 1
    num_dir_face_of_node = g.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dir_bound_faces] = bc_values[external_dir_bound_faces]
    node_val_dir = g.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    node_pressure[is_dir_node] = node_val_dir[is_dir_node]

    # ---------------------- Prepare for exporting ---------------------------

    point_pressures = np.column_stack(
        [node_pressure[nodes_cell], face_pressure[faces_cell]]
    )
    point_coordinates = np.empty([g.dim, g.num_cells, 6])
    point_coordinates[0] = np.column_stack([nx, fx])
    point_coordinates[1] = np.column_stack([ny, fy])

    return point_pressures, point_coordinates


def _oswald_3d(g, g_rot, kw, d):
    # TODO: Implement Oswald interpolator for 3D subdomains
    pass


#%% Pressure postprocessing
def postprocess_pressure(gb, kw, sd_operator_name, p_name):
    """
    Obtain P2 postprocessed coefficients for all nodes of the Grid Bucket.

    Parameters
    ----------
    gb : PorePy object
        Grid Bucket.
    kw : Keyword
        Name of the problem, e.g., "flow"
    sd_operator_name : Keyword
        Subdomain operator keyword, e.g., "diffusion"
    p_name : Keyword
        Subdomain variable, e.g., "pressure"

    Returns
    -------
    None.

    """
    for g, d in gb:

        # Retrieve subdomain discretization
        discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

        # Boolean variable for checking if the scheme is FV
        is_fv = issubclass(type(discr), pp.FVElliptic)

        # Retrieve cell center pressure from the dictionary
        if is_fv:
            pcc = d[pp.STATE][p_name].copy()
        else:
            pcc = discr.extract_pressure(g, d[pp.STATE][p_name], d).copy()

        # Handle the case of 0D domains
        if g.dim == 0:

            d["error_estimates"]["ph"] = pcc

        # Handle the case of 1D, 2D, and 3D domains
        else:

            # Rotate the grid. If g == gb.dim_max() this has no effect
            g_rot = utils.rotate_embedded_grid(g)

            # Retrieve postprocessed coefficients
            coeff = _get_postp_coeff(g, g_rot, d, kw, sd_operator_name, p_name, pcc)

            # Store in data dictionary
            d["error_estimates"]["ph"] = coeff.copy()

    return None


def _get_postp_coeff(g, g_rot, d, kw, sd_operator_name, p_name, pcc):
    """
    Obtain P2 postprocessed for a given node of the Grid bucket.

    Parameters
    ----------
    g : PorePy object
        Grid. It is required that g.dim > 0.
    g_rot : Rotated object
        Rotated pseudo-grid.
    d : Dict
        Data dictionary.
    kw : Keyword
        Name of the problem, e.g., "flow"
    sd_operator_name : Keyword
        Subdomain operator name, e.g., "diffusion"
    p_name : Keyword
        Subdomain variable, e.g., "pressure"
    pcc : NumPy array
        Cell-centered pressure solution.

    Returns
    -------
    coeff : Numpy array
        Coefficients of the post-processed pressure solution. The number of 
        rows are the numbers of cells, and the number of columns corresponds
        to the number of Lagrangian nodes for each element according to its
        dimensionality, i.e., 10 for 3D, 6 for 2D, and 3 for 1D. Note that the
        coefficients are ordered following SciPy ordering convention, i.e., in
        decreasing order starting from the first dimension.

    """

    # Get quadpy elements and cell volumes
    elements = utils._get_quadpy_elements(g, g_rot)
    V = g.cell_volumes

    # Retrieve permeability values
    perm = d[pp.PARAMETERS][kw]["second_order_tensor"].values.copy()
    k_broad = matlib.repmat(perm[0][0], g.dim + 1, 1).T

    # Declare integration methods according to grid dimensionality
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]
    else:
        method = qp.tetrahedron.yu_2()
        int_point = method.points.shape[0]

    # Retrieve reconstructed velocity coefficients
    if "recons_vel" in d["error_estimates"]:
        vel_coeff = d["error_estimates"]["recons_vel"].copy()
    else:
        raise ValueError("Fluxes must be first reconstructed.")

    # Scale by the inverse of the permeability
    # TODO: Handle anisotropy
    vel_coeff /= k_broad

    # Reformat according to quadrature points
    if g.dim == 1:
        a = utils._quadpyfy(vel_coeff[:, 0], int_point)
        b = utils._quadpyfy(vel_coeff[:, 1], int_point)
    elif g.dim == 2:
        a = utils._quadpyfy(vel_coeff[:, 0], int_point)
        b = utils._quadpyfy(vel_coeff[:, 1], int_point)
        c = utils._quadpyfy(vel_coeff[:, 2], int_point)
    else:
        a = utils._quadpyfy(vel_coeff[:, 0], int_point)
        b = utils._quadpyfy(vel_coeff[:, 1], int_point)
        c = utils._quadpyfy(vel_coeff[:, 2], int_point)
        d = utils._quadpyfy(vel_coeff[:, 3], int_point)

    # Declare integrands according to the dimensionality of the grid
    def integrand(X):
        if g.dim == 1:
            x = X
            int_ = -0.5 * a * x ** 2 - b * x
            return int_
        elif g.dim == 2:
            x = X[0]
            y = X[1]
            int_ = -0.5 * a * (x ** 2 + y ** 2) - b * x - c * y
            return int_
        else:
            x = X[0]
            y = X[1]
            z = X[2]
            int_ = -0.5 * a * (x ** 2 + y ** 2 + z ** 2) - b * x - c * y - d * z
            return int_

    # Perform integration
    integral = method.integrate(integrand, elements)

    # Obtain constant of the integration to satisfy |K|^-1 (p_post, 1)_K = p_cc
    # We assume that the postprocessed pressure satisfies (at most) a P2 polynomial
    if g.dim == 1:
        coeff = np.zeros([g.num_cells, 3])
        coeff[:, 0] = -a[:, 0] / 2  # x ** 2
        coeff[:, 1] = -b[:, 0]  # x
        coeff[:, 2] = pcc - integral / V  # 1
    elif g.dim == 2:
        coeff = np.zeros([g.num_cells, 6])
        coeff[:, 0] = -a[:, 0] / 2  # x ** 2
        coeff[:, 1] = np.zeros(g.num_cells)  # x * y
        coeff[:, 2] = -b[:, 0]  # x
        coeff[:, 3] = -a[:, 0] / 2  # y ** 2
        coeff[:, 4] = -c[:, 0]  # y
        coeff[:, 5] = pcc - integral / V  # 1
    elif g.dim == 3:
        coeff = np.zeros([g.num_cells, 10])
        coeff[:, 0] = -a[:, 0] / 2  # x ** 2
        coeff[:, 1] = np.zeros(g.num_cells)  # x * y
        coeff[:, 2] = np.zeros(g.num_cells)  # x * z
        coeff[:, 3] = -b[:, 0]  # x
        coeff[:, 4] = -a[:, 0] / 2  # y ** 2
        coeff[:, 5] = np.zeros(g.num_cells)  # y * z
        coeff[:, 6] = -c[:, 0]  # y
        coeff[:, 7] = -a[:, 0] / 2  # z ** 2
        coeff[:, 8] = -d[:, 0]  # z
        coeff[:, 9] = pcc - integral / V  # 1

    return coeff
