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

import error_estimates_utility as utils

from porepy.grids.grid_bucket import GridBucket


#%% Velocity reconstruction

def subdomain_velocity(gb, kw, lam_name):
    """
    Computes mixed-dimensional flux reconstruction using RT0 extension of 
    normal full fluxes.

    Parameters
    ----------
    gb : PorePy object
        Grid bucket
    g : PorePy object
        Grid
    g_rot : PorePy object
        Rotated grid (for the case of g < gb.dim_max())
    d : dictionary 
        Data dictionary corresponding to the grid g
    kw : Keyword
        Name of the problem
    lam_name : Keyword
        Name of the edge variable
    
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
        
        if g.dim > 0: # flux reconstruction only defined for g.dim > 0
    
            # First, rotate the grid. Note that if g == gb.dim_max(), this has
            # no effect.
            g_rot = utils.rotate_embedded_grid(g)
    
            # Useful mappings
            cell_faces_map, _, _ = sps.find(g.cell_faces)
            cell_nodes_map, _, _ = sps.find(g.cell_nodes())
        
            # Cell-wise arrays
            faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))
            nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
            opp_nodes_cell = utils.get_opposite_side_nodes(g_rot)
            sign_normals_cell = utils.get_sign_normals(g_rot)
            vol_cell = g.cell_volumes
        
            # Opposite side nodes for RT0 extension of normal fluxes
            opp_nodes_coor_cell = np.empty(
                [g_rot.dim, nodes_cell.shape[0], nodes_cell.shape[1]]
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
            full_flux_local_div = (sign_normals_cell * full_flux[faces_cell]).sum(axis=1)
            external_src = d[pp.PARAMETERS][kw]["source"]
            internal_src = _internal_source_term_contribution(gb, g, lam_name)
            np.testing.assert_almost_equal(
                full_flux_local_div,
                external_src + internal_src,
                decimal=7,
                err_msg="Error estimates implemented only for local mass-conservative methods.",
            )
            # ----------------------------------------------------------------
        
            # Perform actual reconstruction and obtain coefficients
            coeffs = np.empty([g_rot.num_cells, g_rot.dim + 1])
            alpha = 1 / (g.dim * vol_cell)
            coeffs[:, 0] = alpha * np.sum(sign_normals_cell * full_flux[faces_cell], axis=1)
            for dim in range(g_rot.dim):
                coeffs[:, dim + 1] = -alpha * np.sum(
                    (sign_normals_cell * full_flux[faces_cell] * opp_nodes_coor_cell[dim]),
                    axis=1,
                )
        
            # --------------------- TEST: Flux reconstruction ----------------
            # Check if the reconstructed evaluated at the face centers normal fluxes
            # match the numerical ones
            recons_flux = _reconstructed_face_fluxes(g_rot, coeffs)
            np.testing.assert_almost_equal(
                recons_flux, full_flux, decimal=12, err_msg="Flux reconstruction has failed."
            )
            # ----------------------------------------------------------------

            # Store coefficients in the data dictionary
            d["error_estimates"]["recons_vel"] = coeffs

    return


def _reconstructed_face_fluxes(g, coeff):
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
        normal_faces_cell[dim] = g.face_normals[dim][faces_cell]
        facenters_cell[dim] = g.face_centers[dim][faces_cell]

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

def subdomain_pressure(gb, sd_operator_name, p_name, p_order):
    """
    Computes subdomain pressure reconstruction for each node of the entire 
    grid bucket according to the prescribed reconstruction order p_order. It is
    assumed that the P0 pressure solution is in d[pp.STATE][p_name], and the 
    nodal pressure in d["error_estimates"][node_pressure].

    Parameters
    ----------
    gb : PorePy object
        Grid Bucket.
    sd_operator_name : Keyword
        Subdomain operator name, i.e., 'difussion'.
    p_name : Keyword
        Subdomain variable name, i.e., 'pressure'
    p_order : Keyword
        Degree of pressure reconstruction, i.e., '1' for P1 elements, or '1.5'
        for P1 elements enriched with bubbles (purely parabolic terms).

    Returns
    -------
    None.
    
    NOTE: The data dictionary will be updated with the field d["error_estimates"]["recons_p"], 
    a Numpy array of shape (g.num_cells x (g.dim + 1)) for P1 elements. and shape
    (g.num_cells x (g.dim + 1)) for P1.5 elements, containing the coefficients
    of the reconstruced pressure field for each subdomain.

    """
    
    for g, d in gb:
        
        # First, rotate the grid. Note that if g == gb.dim_max(), this has
        # no effect.
        g_rot = utils.rotate_embedded_grid(g)
        
        # Retrieve subdomain discretization
        discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

        # Boolean variable for checking is the scheme is FV
        is_fv = issubclass(type(discr), pp.FVElliptic)
        
        # Retrieve cell center pressure
        if is_fv:
            p_cc = d[pp.STATE][p_name]
        else:
            p_cc = discr.extract_pressure(g, d[pp.STATE][p_name], d)
 
        # Retrieve nodal pressures
        if "node_pressure" in d["error_estimates"]:
            p_nv = d["error_estimates"]["node_pressure"]
        else:
            raise("Node pressures must be computed first.")    
 
        # Perform reconstructions according to the degree
        if p_order == '1':
                        
            # Perform  P1 reconstruction    
            coeffs = _P1(g_rot, p_nv)

            # -------------------- TEST: P1 reconstruction -------------------
            recons_pnv, _ = _recons_pnv_and_pcc(g, g_rot, coeffs, p_order)
            np.testing.assert_almost_equal(
                recons_pnv,
                p_nv,
                decimal=12,
                err_msg="P1 pressure reconstruction has failed.",
            )
            # ----------------------------------------------------------------
            
            # Store in data dictionary
            d["error_estimates"]["recons_p"] = coeffs
            
        elif p_order == '1.5':

            # Perform  P1.5 reconstruction    
            coeffs = _P1_plus_bubbles(g_rot, p_nv, p_cc)

            # -------------------- TEST: P1.5 reconstruction -----------------
            recons_pnv, recons_pcc = _recons_pnv_and_pcc(g, g_rot, coeffs, p_order)
            # Test if nodal pressures match reconstructed nodal pressures
            np.testing.assert_almost_equal(
                recons_pnv,
                p_nv,
                decimal=10,
                err_msg="P1.5 pressure reconstruction has failed.",
            )
            # Test if cell center pressures match reconstructed cell center pressures
            np.testing.assert_almost_equal(
                recons_pcc,
                p_cc,
                decimal=10,
                err_msg="P1.5 pressure reconstruction has failed.",
            )
            # ----------------------------------------------------------------
            
            # Store in data dictionary
            d["error_estimates"]["recons_p"] = coeffs
            
        else:
            
            raise("Degree of pressure reconstruction not implemented. Use '1' or '1.5'.")
        
    return 
            

def _P1(g, p_nv):
    """
    Computes pressure reconstruction using P1 elements.

    Parameters
    ----------
    g : PorePy object
        Porepy grid object
    p_nv : NumPy array
        Values of the pressure at the grid nodes.

    Returns
    -------
    coeffs : NumPy array (g.num_cells x (g.dim + 1))
        Coefficients of the reconstructed pressure for each element of the grid.

    
    ------------------------------ TECHNICAL NOTE ----------------------------
    
    The coefficients satisfy the following pressure field depending on the 
    dimensionality of the problem:
         
        p = a,                   (for 0D),
        p = a + bx,              (for 1D),
        p = a + bx + cy,         (for 2D),
        p = a + bx + cy + dz,    (for 3D).
    
    To obtain the coefficients, we solve a linear local problem. For example, 
    for a triangle, we have the node points 1, 2, 3. Then, the local linear 
    system is given by:
  
                  1                              
                 /|       
                / |             |1  x1  y1| |a|   |p1|
               /  |             |1  x2  y2| |b| = |p2| 
              /   |             |1  x3  y3| |c|   |p3|
             /____|       
            3     2       
  
    The extension (or reduction) of the linear system for d != 2 is 
    straightforward.    
    --------------------------------------------------------------------------
    
    """

    #  Handle the case of zero-dimensional grids
    if g.dim == 0:
        return p_nv

    # Retrieving topological data
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodes_coor_cell = np.empty([g.dim, nodes_cell.shape[0], nodes_cell.shape[1]])
    for dim in range(g.dim):
        nodes_coor_cell[dim] = g.nodes[dim][nodes_cell]

    # Assembling the local matrix for each cell
    lcl = np.ones(g.num_cells * (g.dim + 1))
    for dim in range(g.dim):
        lcl = np.column_stack([lcl, nodes_coor_cell[dim].flatten()])
    lcl = np.reshape(lcl, [g.num_cells, g.dim + 1, g.dim + 1])

    # Looping through each element and inverting the local matrix
    coeffs = np.empty([g.num_cells, g.dim + 1])
    vertices_pressures = p_nv[nodes_cell]  # pressure at the vertices
    for cell in range(g.num_cells):
        inv_local_matrix = np.linalg.inv(lcl[cell])
        vert_press_cell = vertices_pressures[cell]
        coeffs[cell] = np.dot(inv_local_matrix, vert_press_cell)

    return coeffs


def _P1_plus_bubbles(g, p_nv, p_cc):
    """
    Computes pressure reconstruction using P1.5 elements. P1.5 elements are
    essentially P1 elements enriched with purely parabolic terms, i.e. x_i**2.
       
    Parameters
    ----------
    g : PorePy object
        Porepy grid object
    p_nv : NumPy array
        Values of the pressure at the grid nodes.
    p_cc : Numpy array
        Values of the pressure at the cell centers.

    Returns
    -------
    coeffs : NumPy array (g.num_cells x (2*g.dim + 1))
        Coefficients of the reconstructed pressure for each element of the grid.    
    
    
    ---------------------------- TECHNICAL NOTE ------------------------------
    
    The coefficients satisfy the following pressure field depending on the 
    dimensionality of the problem:
         
        p = a,                                       (for 0D),
        p = a + bx + ex^2,                           (for 1D),
        p = a + bx + cy + e(x^2 + y^2),              (for 2D),
        p = a + bx + cy + dz + e(x^2 + y^2 + z^2)    (for 3D).
    
    To obtain the coefficients, we solve a linear local problem. For example, 
    for a triangle, we have the node points 1, 2, 3,  and the cell center
    point 4, as shown below. Then the local linear system is given by
  
                  1                              
                 /|       
                / |       |1  x1  y1  x1^2 + y1^2| |a|   |p1|
               /  |       |1  x2  y2  x2^2 + y2^2| |b| = |p2| 
              / 4 |       |1  x3  y3  x3^2 + y3^2| |c|   |p3|
             /____|       |1  x4  y4  x4^2 + y4^2| |e|   |p4|
            3     2       
  
    The extension (or reduction) of the linear system for d != 2 is 
    straightforward.                    
    --------------------------------------------------------------------------

    """

    #  Handle the case of zero-dimensional grids
    if g.dim == 0:
        return p_nv 

    # Retrieving topological data
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodes_coor_cell = np.empty([g.dim, nodes_cell.shape[0], nodes_cell.shape[1]])
    for dim in range(g.dim):
        nodes_coor_cell[dim] = g.nodes[dim][nodes_cell]

    
    # Assembling the local matrix for each cell
    lcl = np.ones(g.num_cells * (g.dim + 2))
    for dim in range(g.dim):
        x = np.column_stack((nodes_coor_cell[dim], g.cell_centers[dim])).flatten()
        lcl = np.column_stack([lcl, x])
        parabola_terms = np.zeros(g.num_cells * (g.dim + 2))
    for dim in range(g.dim):
        x = np.column_stack((nodes_coor_cell[dim], g.cell_centers[dim])).flatten()
        parabola_terms += (x**2)
    lcl = np.column_stack([lcl, parabola_terms])

    # Reshaping
    lcl = np.reshape(lcl, [g.num_cells, g.dim + 2, g.dim + 2])

    # Looping through each element and inverting the local matrix
    coeffs = np.empty([g.num_cells, g.dim + 2])
    vertices_pressures = p_nv[nodes_cell]
    local_pressure = np.column_stack((vertices_pressures, p_cc))
    for cell in range(g.num_cells):
        inv_local_matrix = np.linalg.inv(lcl[cell])
        local_pressure_cell = local_pressure[cell]
        coeffs[cell] = np.dot(inv_local_matrix, local_pressure_cell)

    return coeffs


def _recons_pnv_and_pcc(g, g_rot, coeffs, p_order):
    """
    Evaluate reconstructed pressure a the nodes for testing purposes. 

    Parameters
    ----------
    g : PorePy object
        PorePy grid object.
    g_rot : PorePy object
        PorePy << rotated >> grid object.
    coeffs : NumPy array (with the appropiate dimension according to p_order)
        Coefficients of the reconstructed pressure field
    p_order : Keyword
        Pressure reconstruction order, i.e., '1' or '1.5'

    Returns
    -------
    recons_pnv : NumPy array (g.num_nodes)
        Reconstructed pressure evaluated at the nodes.
        
    recons_pcc: NumPy array (g.num_cells)
        Reconstructed pressure evaluated at the cell centers.    

    """

    # Handle the case of zero-dimensional grids
    if g.dim == 0:
        rec_pnv = coeffs
        return rec_pnv

    # Useful mapping
    nc = g.num_cells
    nn = g.num_nodes
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodes_coor_cell = np.empty([g.dim, nc, g.dim + 1])
    for dim in range(g.dim):
        nodes_coor_cell[dim] = g_rot.nodes[dim][nodes_cell]
    
    # Compute P1 reconstructed nodal pressures
    rec_pnv = matlib.repmat(coeffs[:, 0].reshape([nc, 1]), 1, g.dim + 1)
    for dim in range(g.dim):
        rec_pnv += matlib.repmat(coeffs[:, dim+1].reshape([nc, 1]), 1, g.dim + 1) * nodes_coor_cell[dim]

    # For P1.5, add the bubbles contribution
    if p_order == '1.5':
        for dim in range(g.dim):
            rec_pnv += matlib.repmat(coeffs[:, -1].reshape([nc, 1]), 1, g.dim + 1) * nodes_coor_cell[dim]**2
        
    # Flattening and formating
    idx =  np.array([np.where(nodes_cell.flatten() == x)[0][0] for x in np.arange(nn)])
    rec_pnv_flat = rec_pnv.flatten()
    recons_pnv = rec_pnv_flat[idx]

    # Also, obtain cell-center reconstructed pressures for P1.5
    recons_pcc = coeffs[:, 0]
    for dim in range(g.dim):
        recons_pcc += (coeffs[:, dim+1] * g_rot.cell_centers[dim] +
                            coeffs[:, -1] * g_rot.cell_centers[dim] ** 2)
       
    return recons_pnv, recons_pcc
    
    
#%% Mortar flux reconstruction
    
    