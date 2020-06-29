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
from typing import Union

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

    return None


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
    g_rot : Rotated object
        Rotated PorePy pseudo-grid.
        
    """

    class RotatedGrid:
        """
        This class creates a rotated grid object. 
        """

        def __init__(
            self,
            cell_centers,
            face_normals,
            face_centers,
            rotation_matrix,
            dim_bool,
            nodes,
        ):

            self.cell_centers = cell_centers
            self.face_normals = face_normals
            self.face_centers = face_centers
            self.rotation_matrix = rotation_matrix
            self.dim_bool = dim_bool
            self.nodes = nodes

        def __str__(self):
            return "Rotated pseudo-grid object"

        def __repr__(self):
            return (
                "Rotated pseudo-grid object with atributes:\n"
                + "cell_centers\n"
                + "face_normals\n"
                + "face_centers\n"
                + "rotation_matrix\n"
                + "dim_bool\n"
                + "nodes"
            )

    # Rotate grid
    (
        cell_centers,
        face_normals,
        face_centers,
        rotation_matrix,
        dim_bool,
        nodes,
    ) = pp.map_geometry.map_grid(g)

    # Create rotated grid object
    rotated_grid = RotatedGrid(
        cell_centers, face_normals, face_centers, rotation_matrix, dim_bool, nodes
    )

    return rotated_grid


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


def get_sign_normals(g, g_rot):
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
        faceCntr_cells[dim] = g_rot.face_centers[dim][faces_cell]

    # Global normals of the faces per cell
    glb_normal_faces_cell = np.empty([g.dim, faces_cell.shape[0], faces_cell.shape[1]])
    for dim in range(g.dim):
        glb_normal_faces_cell[dim] = g_rot.face_normals[dim][faces_cell]

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
        cellCntr_broad[dim] = matlib.repmat(g_rot.cell_centers[dim], g.dim + 1, 1).T

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

def _get_quadpy_elements(g, g_rot):
    """
    Assembles the elements of a given grid in quadpy format
    For a 2D example see: https://pypi.org/project/quadpy/

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object.
        
    Returns
    -------
    quadpy_elements : NumPy array
        Elements in QuadPy format.

    Example
    -------
    >>> # shape (3, 5, 2), i.e., (corners, num_triangles, xy_coords)
    >>> triangles = numpy.stack([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.2, 0.6], [1.3, 0.7], [1.4, 0.8]],
            [[26.0, 31.0], [24.0, 27.0], [33.0, 28]],
            [[0.1, 0.3], [0.4, 0.4], [0.7, 0.1]],
            [[8.6, 6.0], [9.4, 5.6], [7.5, 7.4]]
            ], axis=-2)
    
    """

    # Renaming variables
    nc = g.num_cells

    # Getting node coordinates for each cell
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([nc, g.dim + 1]))
    nodes_coor_cell = np.empty([g.dim, nodes_cell.shape[0], nodes_cell.shape[1]])
    for dim in range(g.dim):
        nodes_coor_cell[dim] = g_rot.nodes[dim][nodes_cell]

    # Stacking node coordinates
    cnc_stckd = np.empty([nc, (g.dim + 1) * g.dim])
    col = 0
    for vertex in range(g.dim + 1):
        for dim in range(g.dim):
            cnc_stckd[:, col] = nodes_coor_cell[dim][:, vertex]
            col += 1
    element_coord = np.reshape(cnc_stckd, np.array([nc, g.dim + 1, g.dim]))

    # Reshaping to please quadpy format i.e, (corners, num_elements, coords)
    elements = np.stack(element_coord, axis=-2)

    # For some reason, quadpy needs a different formatting for line segments
    if g.dim == 1:
        elements = elements.reshape(g.dim + 1, g.num_cells)

    return elements


def _quadpyfy(array, integration_points):
    """
    Format array for numerical integration

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    integration_points : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return matlib.repmat(array, integration_points, 1).T


def _find_edges(
    g: pp.Grid, loc_faces: np.ndarray, central_node: int
) -> Union[np.ndarray, np.ndarray]:
    """
    Find the 1d edges around a central node in a 3d grid.
    Args:
        g (pp.Grid): Macro grid.
        loc_faces (np.ndarray): Index of faces that have central_node among their
            vertexes.
        central_node (int): Index of the central node.
    Returns:
        nodes_on_edges (np.ndarray): Index of nodes that form a 1d edge together with
            the central node.
        face_of_edges (np.ndarray): Faces corresponding to the edge.
    Raises:
        ValueError: If not all faces in the grid have the same number of nodes.
    """

    fn_loc = g.face_nodes[:, loc_faces]
    node_ind = fn_loc.indices
    fn_ptr = fn_loc.indptr

    if not np.unique(np.diff(fn_ptr)).size == 1:
        # Fixing this should not be too hard
        raise ValueError("Have not implemented grids with varying number of face-nodes")

    # Number of nodes per face
    num_fn = np.unique(np.diff(fn_ptr))[0]

    # Sort the nodes of the local faces.
    sort_ind = np.argsort(node_ind)

    # The elements in sorted_node_ind (and node_ind) will not be unique
    sorted_node_ind = node_ind[sort_ind]

    # Duplicate the face indices, and make the same sorting as for the nodes
    face_ind = np.tile(loc_faces, (num_fn, 1)).ravel(order="f")
    sorted_face_ind = face_ind[sort_ind]

    # Exclude nodes and faces that correspond to the central node
    not_central_node = np.where(sorted_node_ind != central_node)[0]
    sorted_node_ind_ext = sorted_node_ind[not_central_node]
    sorted_face_ind_ext = sorted_face_ind[not_central_node]

    # Nodes that occur more than once are part of at least two faces, thus there is an
    # edge going from the central to that other node
    # This may not be true for sufficiently degenerated grids (not sure what that means,
    # possibly something with hanging nodes).
    multiple_occur = np.where(np.bincount(sorted_node_ind_ext) > 1)[0]
    hit = np.in1d(sorted_node_ind_ext, multiple_occur)

    # Edges (represented by the node that is not the central one), and the faces of the
    # edges. Note that neither nodes_on_edges nor face_of_edges are unique, however, the
    # combination of an edge and a face should be so.
    nodes_on_edges = sorted_node_ind_ext[hit]
    face_of_edges = sorted_face_ind_ext[hit]

    return nodes_on_edges, face_of_edges


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

        d["error_estimates"] = {}

    for e, d_e in gb.edges():

        d_e["error_estimates"] = {}

    return


def transfer_error_to_state(gb):

    for g, d in gb:

        if g.dim == 0:
            d[pp.STATE]["diffusive_error"] = None
            d[pp.STATE]["nonconf_error"] = None
        else:
            if "diffusive_error" and "nonconf_error" in d["error_estimates"]:
                d[pp.STATE]["diffusive_error"] = d["error_estimates"]["diffusive_error"]
                d[pp.STATE]["nonconf_error"] = d["error_estimates"]["nonconf_error"]
            else:
                raise ValueError("Error estimates must be computed first.")

    for _, d_e in gb.edges():

        g_m = d_e["mortar_grid"]

        if g_m.dim == 0:
            d[pp.STATE]["diffusive_error"] = None
            d[pp.STATE]["nonconf_error"] = None
        else:
            if "diffusive_error" and "nonconf_error" in d_e["error_estimates"]:
                d_e[pp.STATE]["diffusive_error"] = d_e["error_estimates"][
                    "diffusive_error"
                ]
                d_e[pp.STATE]["nonconf_error"] = d_e["error_estimates"]["nonconf_error"]
            else:
                raise ValueError("Error estimates must be computed first.")


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

    for g, d in gb:
        if g.dim > 0:
            global_error += d["error_estimates"]["diffusive_error"].sum()
            global_error += d["error_estimates"]["nonconf_error"].sum()

    for _, d_e in gb.edges():

        g_m = d_e["mortar_grid"]

        if g_m.dim > 0:

            global_error += d_e["error_estimates"]["diffusive_error"].sum()
            global_error += d_e["error_estimates"]["nonconf_error"].sum()

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

    # Handle the case of zero-dimensional grids
    if g.dim == 0:
        raise ValueError("Error estimates are not defined for zero-dimensional grids")

    subdomain_error = d["error_estimates"]["diffusive_error"].sum()
    subdomain_error += d["error_estimates"]["nonconf_error"].sum()

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

    interface_error = d["error_estimates"]["diffusive_error"].sum()
    interface_error += d["error_estimates"]["nonconf_error"].sum()

    return interface_error
