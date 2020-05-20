#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:27:19 2020

@author: jv
"""

import numpy as np
import numpy.matlib as matlib
import porepy as pp
import scipy.sparse as sps

from porepy.grids.grid_bucket import GridBucket


#%% Flux related functions
def compute_full_flux(gb, kw, sd_operator_name, p_name, lam_name):
    """
    Computes full flux for the entire grid bucket. The full flux is composed 
    of the subdomain Darcy flux, plus the projection of the lower dimensional 
    neighboring interface (mortar) fluxes associated with such subdomain.

    Parameters
    ----------
    gb : PorePy Object
        GridBucket object.
    kw : Keyword
        Problem keyword, i.e., flow.
    sd_operator_name : Keyword
        Subdomain operator name, i.e., diffusion.
    p_name : Keyword
        Subdomain variable name, i.e., pressure.
    lam_name : Keyword
        Edge variable, i.e., mortar solution.

    Returns
    -------
    None. 
    
    Note: The data dictionary of each subdomain is updated with the field 
    d["error_estimates"]["full_flux"], a NumPy array of length g.num_faces.

    """

    # Loop through all the nodes from the grid bucket
    for g, d in gb:

        if g.dim > 0:  # full-flux only makes sense for g.dim > 0

            # Retrieve subdomain discretization
            discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

            # Boolean variable for checking is the scheme is FV
            is_fv = issubclass(type(discr), pp.FVElliptic)

            if is_fv:  # fvm-schemes

                # Compute Darcy flux
                parameter_dictionary = d[pp.PARAMETERS][kw]
                matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][kw]
                darcy_flux = (
                    matrix_dictionary["flux"] * d[pp.STATE][p_name]
                    + matrix_dictionary["bound_flux"]
                    * parameter_dictionary["bc_values"]
                )

                # Add the contribution of the mortar fluxes for each edge associated
                # to the higher-dimensional subdomain g
                induced_flux = np.zeros(darcy_flux.size)
                faces = g.tags["fracture_faces"]
                if np.any(faces):

                    for _, d_e in gb.edges_of_node(g):
                        g_m = d_e["mortar_grid"]
                        if g_m.dim == g.dim:
                            continue
                        # project the mortar variable back to the higher dimensional
                        # subdomain
                        induced_flux += (
                            matrix_dictionary["bound_flux"]
                            * g_m.mortar_to_master_int()
                            * d_e[pp.STATE][lam_name]
                        )

                # Store in data dictionary
                d["error_estimates"]["full_flux"] = darcy_flux + induced_flux

            else:  # fem-schemes

                # Retrieve Darcy flux from the solution array
                darcy_flux = discr.extract_flux(g, d[pp.STATE][p_name], d)

                # We need to recover the flux from the mortar variable before
                # the projection, only lower dimensional edges need to be considered.
                induced_flux = np.zeros(darcy_flux.size)
                faces = g.tags["fracture_faces"]
                if np.any(faces):
                    # recover the sign of the flux, since the mortar is assumed
                    # to point from the higher to the lower dimensional problem
                    _, indices = np.unique(g.cell_faces.indices, return_index=True)
                    sign = sps.diags(g.cell_faces.data[indices], 0)

                    for _, d_e in gb.edges_of_node(g):
                        g_m = d_e["mortar_grid"]
                        if g_m.dim == g.dim:
                            continue
                        # project the mortar variable back to the higher dimensional
                        # subdomain
                        induced_flux += (
                            sign
                            * g_m.master_to_mortar_avg().T
                            * d_e[pp.STATE][lam_name]
                        )

                # Store in data dictionary
                d["error_estimates"]["full_flux"] = darcy_flux + induced_flux

    return

#%% Pressure related functions
def compute_node_pressure(gb, kw, sd_operator_name, p_name, nodal_method):
    """
    This function simply calls either the 'flux-inverse' or the 'k-averaging'
    nodal interpolation methods, for each node of the grid bucket.

    Parameters
    ----------
    gb : PorePy object
        Grid bucket, with the solution computed in d[pp.STATE][p_name]
    kw : Keyword
        Problem name, i.e., 'flow'.
    sd_operator_name : Keyword
        Subdomain operator name, i.e., 'difussion'.
    p_name : Keyword
        Subdomain variable name, i.e., 'pressure'.
    nodal_method : Keyword
        Nodal interpolation method, i.e., 'flux-inverse' or 'k-averaging'

    Returns
    -------
    None.

    """
    
    for g, d in gb:
        
        if nodal_method == 'flux-inverse':
            _compute_node_pressure_invflux(g, d, kw, sd_operator_name, p_name)
        elif nodal_method == "k-averaging":
            _compute_node_pressure_kavg(g, d, kw, sd_operator_name, p_name)
        else:
            raise("Nodal pressure interpolation method not implemented. Use either 'flux-inverse' or 'k-averaging'.")

    return


def _compute_node_pressure_invflux(g, d, kw, sd_operator_name, p_name):
    """
    Computes nodal pressure values using the inverse of the flux

    Parameters
    ----------
    g : PorePy object
        Grid, i.e., node from the grid bucket
    d : dictionary 
        Data dictionary
    kw : Keyword
        Problem name, i.e., 'flow'.
    p_name : Keyword
        Subdomain variable name, i.e., 'pressure'.

    Returns
    -------
    None.
    
    Note: The data dictionary will be updated with  d["error_estimates"]["node_pressure"],
    a NumPy array of length g.num_nodes.

    """
    
    # Retrieve subdomain discretization
    discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

    # Boolean variable for checking is the scheme is FV
    is_fv = issubclass(type(discr), pp.FVElliptic)
    
    # Retrieve pressure from the dictionary
    if is_fv:
        p = d[pp.STATE][p_name]
    else:
        p = discr.extract_pressure(g, d[pp.STATE][p_name], d)
        
    # Handle the case of zero-dimensional grids
    if g.dim == 0:
        d["error_estimates"]["node_pressure"] = p
        return

    # Retrieve topological data
    nc = g.num_cells
    nf = g.num_faces
    nn = g.num_nodes

    # Retrieve subdomain discretization
    discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

    # Perform reconstruction [This is Eirik's original implementation]
    cell_nodes = g.cell_nodes()
    cell_node_volumes = cell_nodes * sps.dia_matrix((g.cell_volumes, 0), (nc, nc))
    sum_cell_nodes = cell_node_volumes * np.ones(nc)
    cell_nodes_scaled = (
        sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes
    )

    # Retrieving numerical full fluxes
    if "full_flux" in d["error_estimates"]:
        flux = d["error_estimates"]["full_flux"]
    else:
        raise('Full flux must be computed first')

    # Project fluxes
    proj_flux = pp.RT0(kw).project_flux(g, flux, d)[: g.dim]

    # Obtaining local gradients
    loc_grad = np.zeros((g.dim, nc))
    perm = d[pp.PARAMETERS][kw]["second_order_tensor"].values
    for ci in range(nc):
        loc_grad[: g.dim, ci] = -np.linalg.inv(perm[: g.dim, : g.dim, ci]).dot(
            proj_flux[:, ci]
        )
        
    # Obtaining nodal pressures
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    cell_node_matrix = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodal_pressures = np.zeros(nn)

    for col in range(g.dim + 1):
        nodes = cell_node_matrix[:, col]
        dist = g.nodes[: g.dim, nodes] - g.cell_centers[: g.dim]
        scaling = cell_nodes_scaled[nodes, np.arange(nc)]
        contribution = (
            np.asarray(scaling)
            * (p + np.sum(dist * loc_grad, axis=0))
        ).ravel()
        nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)

    # Treatment of boundary conditions
    bc = d[pp.PARAMETERS][kw]["bc"]
    bc_values = d[pp.PARAMETERS][kw]["bc_values"]

    external_dirichlet_boundary = np.logical_and(
        bc.is_dir, g.tags["domain_boundary_faces"]
    )

    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = g.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_values[external_dirichlet_boundary]

    node_val_dir = g.face_nodes * face_vec

    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

    # Save in the dictionary
    d["error_estimates"]["node_pressure"] = nodal_pressures

    return


def _compute_node_pressure_kavg(g, d, kw, sd_operator_name, p_name):
    """
    Computes nodal pressure values using an average of the cell permeabilities
    associated with the node patch.

    Parameters
    ----------
    g : PorePy object
        Grid, i.e., node from the grid bucket
    d : dictionary 
        Data dictionary
    kw : Keyword
        Problem name, i.e., 'flow'.
    p_name : Keyword
        Subdomain variable name, i.e., 'pressure'.

    Returns
    -------
    None.
    
    Note: The data dictionary will be updated with the field d["error_estimates"]["node_pressure"]

    """

    # Retrieve subdomain discretization
    discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

    # Boolean variable for checking is the scheme is FV
    is_fv = issubclass(type(discr), pp.FVElliptic)
    
    # Retrieve pressure from the dictionary
    if is_fv:
        p_cc = d[pp.STATE][p_name]
    else:
        p_cc = discr.extract_pressure(g, d[pp.STATE][p_name], d)
        
    # Handle the case of zero-dimensional grids
    if g.dim == 0:
        d["error_estimates"]["node_pressure"] = p_cc
        return

    # Topological data
    nn = g.num_nodes
    nf = g.num_faces
    V = g.cell_volumes
    
    # Retrieve permeability values
    k = d[pp.PARAMETERS][kw]["second_order_tensor"].values
    perm = k[0][0]
    # TODO: For the moment, we assume kxx = kyy = kzz on each cell
    # It would be nice to add the possibility to account for anisotropy

    nodes_of_cell, cell_idx, _ = sps.find(g.cell_nodes()) 
    p_contribution = p_cc[cell_idx]
    k_contribution = perm[cell_idx]
    V_contribution = V[cell_idx]
    
    numer_contribution = p_contribution * k_contribution * V_contribution
    denom_contribution = k_contribution * V_contribution
    
    numer = np.bincount(nodes_of_cell, weights=numer_contribution, minlength=nn)
    denom = np.bincount(nodes_of_cell, weights=denom_contribution, minlength=nn)
    
    nodal_pressures = numer / denom
    
    #Deal with Dirichlet and Neumann boundary conditions
    bc = d[pp.PARAMETERS][kw]["bc"]
    bc_values = d[pp.PARAMETERS][kw]["bc_values"]
    external_dirichlet_boundary = np.logical_and(
        bc.is_dir, g.tags["domain_boundary_faces"]
    )
    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = g.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_values[external_dirichlet_boundary]
    node_val_dir = g.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

    # Save in the dictionary
    d["error_estimates"]["node_pressure"] = nodal_pressures
    
    return

#%% Mortar related functions
def compute_node_mortar_flux(g_m, d_e, lam_name):
    
    V = g_m.cell_volumes()
    
    
    pass

#%% Geometry related functions
def rotate_embedded_grid(g):
    """
    Rotates grid to account for embedded fractures. 
    
    Note that the pressure and flux reconstruction use the rotated grids, 
    where only the relevant dimensions are taken into account, e.g., a 
    one-dimensional tilded fracture will be represented by a three-dimensional 
    grid, where only the first dimension is used.
    
    Parameters
    ----------
    g : PorePy object
        Original (unrotated) PorePy grid.

    Returns
    -------
    g_rot : Porepy object
        Rotated PorePy grid.
        
    """

    # Copy grid to keep original one untouched
    g_rot = g.copy()

    # Rotate grid
    (
        cell_centers,
        face_normals,
        face_centers,
        R,
        dim,
        nodes,
    ) = pp.map_geometry.map_grid(g_rot)

    # Update rotated fields in the relevant dimension
    for dim in range(g.dim):
        g_rot.cell_centers[dim] = cell_centers[dim]
        g_rot.face_normals[dim] = face_normals[dim]
        g_rot.face_centers[dim] = face_centers[dim]
        g_rot.nodes[dim] = nodes[dim]

    return g_rot


def get_opposite_side_nodes(g):
    """
    Computes opposite side nodes for each face of each cell in the grid

    Parameters
    ----------
    g : PorePy object
        Porepy grid object. Note that for mixed-dimensional grids, this will
        correspond to the rotated grid object.

    Returns
    -------
    opposite_nodes : NumPy array (cell_numbers x (g.dim + 1))
        Rows represent the cell number and the columns represent the opposite 
        side node index of the face.
    
    """

    # Retrieving toplogical data
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    face_nodes_map, _, _ = sps.find(g.face_nodes)
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())

    faces_of_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodes_of_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodes_of_face = face_nodes_map.reshape((np.array([g.num_faces, g.dim])))

    opposite_nodes = np.empty_like(faces_of_cell)
    for cell in range(g.num_cells):
        opposite_nodes[cell] = [
            np.setdiff1d(nodes_of_cell[cell], nodes_of_face[face])
            for face in faces_of_cell[cell]
        ]

    return opposite_nodes


def get_sign_normals(g):
    """
    Computes sign of the face normals for each element in the grid

    Parameters
    ----------
    g : PorePy object
        Porepy grid object. Note that for mixed-dimensional grids, this will
        correspond to the rotated grid object.

    Returns
    -------
    sign_normals : NumPy array
        A value equal to 1 implies that the sign of the basis remain unchanged
        A value equal to -1 implies that the sign of the basis needs to be flipped
    
    """

    # We have to take care of the sign of the basis functions. The idea
    # is to create an array of signs "sign_normals" that will be multiplying
    # each edge basis function.
    # To determine this array, we need the following:
    #   (1) Compute the local outter normal (lon) vector for each cell
    #   (2) For every face of each cell, compare if lon == global normal vector.
    #       If they're not, then we need to flip the sign of lon for that face

    # Faces associated to each cell
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

    # Face centers coordinates for each face associated to each cell
    faceCntr_cells = np.empty([g.dim, faces_cell.shape[0], faces_cell.shape[1]])
    for dim in range(g.dim):
        faceCntr_cells[dim] = g.face_centers[dim][faces_cell]

    # Global normals of the faces per cell
    glb_normal_faces_cell = np.empty([g.dim, faces_cell.shape[0], faces_cell.shape[1]])
    for dim in range(g.dim):
        glb_normal_faces_cell[dim] = g.face_normals[dim][faces_cell]

    # Computing the local outter normals of the faces per cell.
    # To do this, we first assume that n_loc = n_glb, and then we fix the sign.
    # To fix the sign, we compare the length of two vectors,
    # the first vector v1 = face_center - cell_center, and the second vector v2
    # is a prolongation of v1 in the direction of the normal. If ||v2||<||v1||,
    # then the normal of the face in question is pointing inwards, and we needed
    # to flip the sign.
    loc_normal_faces_cell = glb_normal_faces_cell.copy()
    cellCntr_broad = np.empty([g.dim, g.num_cells, g.dim + 1])
    for dim in range(g.dim):
        cellCntr_broad[dim] = matlib.repmat(g.cell_centers[dim], g.dim + 1, 1).T

    v1 = faceCntr_cells - cellCntr_broad
    v2 = v1 + loc_normal_faces_cell * 0.001

    # Checking if ||v2|| < ||v1|| or not
    length_v1 = np.linalg.norm(v1, axis=0)
    length_v2 = np.linalg.norm(v2, axis=0)
    swap_sign = 1 - 2 * np.int8(length_v2 < length_v1)
    # Swapping the sign of the local normal vectors
    loc_normal_faces_cell *= swap_sign

    # Now that we have the local outter normals. We can check if the local
    # and global normals are pointing in the same direction. To do this
    # we compute lenght_sum_n = || n_glb + n_loc||. If they're the same, then
    # lenght_sum_n > 0. Otherwise, they're opposite and lenght_sum_n \approx 0.
    sum_n = loc_normal_faces_cell + glb_normal_faces_cell
    length_sum_n = np.linalg.norm(sum_n, axis=0)
    sign_normals = 1 - 2 * np.int8(length_sum_n < 1e-8)

    return sign_normals

#%% Miscelaneaous functions
def init_estimates_data_keyword(gb):
    """
    Utility function that initializes the keyword ["error_estimates"] inside
    the data dictionary for all nodes and edges of the grid bucket

    Parameters
    ----------
    gb : PorePy object
        Grid Bucket.

    Returns
    -------
    None.

    """
    
    for g, d in gb:
        
        d["error_estimates"] = { }
        
    for e, d_e in gb.edges():
        
        d_e["error_estimates"] = { }
        
    return


#%% Global errors related functions
def compute_global_error(gb, data=None):
    """
    Computes the sum of the local errors by looping through all the subdomains
    and interfaces. In the case of mono-dimensional grids, the grid and the data
    dictionary must be passed.

    Parameters
    ----------
    gb : PorePy object
        PorePy grid bucket object. Alternatively, g for mono-dimensional grids.
    
    data : Dictionary
        Dicitonary of the mono-dimensional grid. This field is not used for the
        mixed-dimensional case.

    Returns
    -------
    global_error : Scalar
        Global error, i.e., sum of local errors.

    """
    # TODO: Check first if the estimate is in the dictionary, if not, throw an error

    global_error = 0

    # Obtain global error for mono-dimensional grid
    if not isinstance(gb, GridBucket) and not isinstance(gb, pp.GridBucket):
        global_error = data[pp.STATE]["error_DF"].sum()
    
    # Obtain global error for mixed-dimensional grids
    else:
        for g, d in gb:
            if g.dim > 0:
                global_error += d[pp.STATE]["error_DF"].sum()

        for e, d in gb.edges():
            # TODO: add diffusive flux error contribution for the interfaces
            # global_error += d[]
            pass

    return global_error


def compute_subdomain_error(g, d):
    """
    Computes the sum of the local errors for a specific subdomain.

    Parameters
    ----------
    g : Grid
        DESCRIPTION.
    d : Data dictionary
        DESCRIPTION.

    Returns
    -------
    subdomain_error : TYPE
        DESCRIPTION.

    """

    subdomain_error = d[pp.STATE]["error_DF"].sum()

    return subdomain_error


def compute_interface_error(g, d):
    """
    Computes the sum of the local errors for a specific interface.

    Parameters
    ----------
    g : Mortar grid
        DESCRIPTION.
    d : Data dictionary
        DESCRIPTION.

    Returns
    -------
    subdomain_error : TYPE
        DESCRIPTION.

    """

    interface_error = d[pp.STATE]["error_DF"].sum()

    return interface_error
    