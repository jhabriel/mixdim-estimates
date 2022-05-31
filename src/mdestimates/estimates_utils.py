from __future__ import annotations
import porepy as pp
import numpy as np
import numpy.matlib as matlib
import mdestimates as mde
import scipy.sparse as sps

from typing import List, Tuple, Union


def get_opposite_side_nodes(g: pp.Grid) -> np.ndarray:
    """
    Computes opposite side nodes for each face of each cell in the grid.

    Parameters
    -----------
        g (pp.Grid): PorePy grid.

    Returns
    --------
        opposite_nodes (np.ndarray): Opossite nodes with rows representing the cell number and
            columns representing the opposite side node index of the face. The size of the
            array is (g.num_cells x (g.dim + 1)).
    """

    # Retrieving toplogical data
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    face_nodes_map, _, _ = sps.find(g.face_nodes)
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())

    # Reshape maps
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


def get_sign_normals(g: pp.Grid, g_rot: mde.RotatedGrid) -> np.ndarray:
    """
    Computes sign of the face normals for each cell of the grid.

    Parameters
    ----------
        g (pp.Grid): PorePy grid.
        g_rot (mde.RotatedGrid): Rotated pseudo-grid.

    Returns
    -------
    sign_normals (np.ndarray): Sign of the face normal. 1 if the signs of the local and global
        normals are the same, -1 otherwise. The size of the array is g.num_faces.
    """
    # We have to take care of the sign of the basis functions. The idea is to create an array
    # of signs "sign_normals" that will be multiplying each edge basis function for the RT0
    # reconstruction of fluxes.
    # To determine this array, we need the following:
    #   (1) Compute the local outter normal (lon) vector for each cell
    #   (2) For every face of each cell, compare if lon == global normal vector.
    #       If they're not, then we need to flip the sign of lon for that face

    # Faces associated to each cell
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

    # Face centers coordinates for each face associated to each cell
    face_center_cells = g_rot.face_centers[:, faces_cell]

    # Global normals of the faces per cell
    global_normal_faces_cell = g_rot.face_normals[:, faces_cell]

    # Computing the local outter normals of the faces per cell. To do this, we first assume
    # that n_loc = n_glb, and then we fix the sign. To fix the sign, we compare the length
    # of two vectors, the first vector v1 = face_center - cell_center, and the second vector v2
    # is a prolongation of v1 in the direction of the normal. If ||v2||<||v1||, then the
    # normal of the face in question is pointing inwards, and we needed to flip the sign.
    local_normal_faces_cell = global_normal_faces_cell.copy()
    cell_center_broadcast = np.empty([g.dim, g.num_cells, g.dim + 1])
    for dim in range(g.dim):
        cell_center_broadcast[dim] = matlib.repmat(g_rot.cell_centers[dim], g.dim + 1, 1).T

    v1 = face_center_cells - cell_center_broadcast
    v2 = v1 + local_normal_faces_cell * 0.001

    # Checking if ||v2|| < ||v1|| or not
    length_v1 = np.linalg.norm(v1, axis=0)
    length_v2 = np.linalg.norm(v2, axis=0)
    swap_sign = 1 - 2 * np.int8(length_v2 < length_v1)
    # Swapping the sign of the local normal vectors
    local_normal_faces_cell *= swap_sign

    # Now that we have the local outter normals. We can check if the local
    # and global normals are pointing in the same direction. To do this
    # we compute lenght_sum_n = || n_glb + n_loc||. If they're the same, then
    # lenght_sum_n > 0. Otherwise, they're opposite and lenght_sum_n \approx 0.
    sum_n = local_normal_faces_cell + global_normal_faces_cell
    length_sum_n = np.linalg.norm(sum_n, axis=0)
    sign_normals = 1 - 2 * np.int8(length_sum_n < 1e-8)

    return sign_normals


def get_quadpy_elements(g: pp.Grid, g_rot: mde.RotatedGrid) -> np.ndarray:
    """
    Assembles the elements of a given grid in quadpy format: https://pypi.org/project/quadpy/.

    Parameters
    ----------
        g (pp.Grid): PorePy grid.
        g_rot (mde.RotatedGrid): Rotated pseudo-grid.

    Returns
    --------
    quadpy_elements (np.ndarray): Elements in QuadPy format.

    Example
    --------
    >>> # shape (3, 5, 2), i.e., (corners, num_triangles, xy_coords)
    >>> triangles = np.stack([
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
    nodes_coor_cell = g_rot.nodes[:, nodes_cell]

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


def get_qp_elements_from_union_grid_1d(union_grid: np.ndarray) -> np.ndarray:
    """
    Assembles the elements of a one-dimensional union grid in QuadPy format.

    Parameters:
    -----------
        union_grid (np.ndarray): Union grid generated from two non-mathching grids.

    Returns:
    --------
        elements (np.ndarray): Cells arranged in QuadPy format.
    """
    nc = union_grid.shape[0]
    dim = 1

    # Stacking node coordinates
    cnc_stckd = np.empty([nc, (dim + 1) * dim])
    col = 0
    for vertex in range(dim + 1):
        cnc_stckd[:, col] = union_grid[:, vertex]
        col += 1
    element_coord = np.reshape(cnc_stckd, np.array([nc, dim + 1, dim]))

    # Reshaping to please quadpy format i.e, (corners, num_elements, coords)
    elements = np.stack(element_coord, axis=-2)

    # For some reason, QuadPy needs a different formatting for line segments
    if dim == 1:
        elements = elements.reshape(dim + 1, nc)

    return elements


def cell_edge_map(g: pp.Grid) -> np.ndarray:
    """
    Compute cell-edges mapping for 3D grids

    Parameters
    ----------
    g (pp.Grid): PorePy grid with g.dim = 3 and computed geometry.

    Returns
    -------
    cell_edge (ndarray): cell-edge mapping with size 2 x num_cells x 6. The number 2 comes
        from the the fact that an edge is defined by a pair of nodes, and 6 are total number
        edges for a tetrahedra.

    NOTE: The mapping is constructed such that the first dimension refers to the node number,
    the second dimension refers to the cell number, and the third dimension refers to the
    edge number. For example, the pair of nodes defining the edge number 4 of the cell
    number 2 is given by the tuple: (cell_edge[0][2][4], cell_edge[1][2][4]).

    """

    # Sanity check
    if g.dim != 3:
        raise ValueError("cell_edge_map only makes sense for three-dimensional grids")

    # Retrieve number of cells and nodes for rapid access
    nc: int = g.num_cells
    nn: int = g.num_nodes

    # Obtain mappings in dense form

    # node_of_cell is a matrix with "nodes" as rows and "cells" as columns. For example,
    # nodes_of_cell[0] will return the (four) node numbers associated to the cell number 0
    nodes_of_cell: np.ndarray = sps.find(g.cell_nodes().T.A)[1].reshape((nc, 4))

    # cells_of_node is the transpose of node_of_cell
    cells_of_node: np.ndarray = nodes_of_cell.T

    # Now we have to "manually" create the edges. Since a tetrahedra has 4 nodes,
    # each cell will contain six edges
    edges = np.empty((6, 2, nc), dtype=np.int32)
    edges[0] = np.vstack([cells_of_node[0], cells_of_node[1]])
    edges[1] = np.vstack([cells_of_node[0], cells_of_node[2]])
    edges[2] = np.vstack([cells_of_node[0], cells_of_node[3]])
    edges[3] = np.vstack([cells_of_node[1], cells_of_node[2]])
    edges[4] = np.vstack([cells_of_node[1], cells_of_node[3]])
    edges[5] = np.vstack([cells_of_node[2], cells_of_node[3]])

    # Sort in ascending order the pair of nodes defining an edge. E.g., the pair (1, 0) will
    # be changed to (0, 1).
    edges_sorted: np.ndarray = np.empty((6, 2, nc), dtype=np.int32)
    for edge in range(6):
        edges_sorted[edge] = np.sort(edges[edge], axis=0)

    # Create a list of edges
    edge_list = []
    for edge in range(6):
        for cell in range(nc):
            edge_list.append(tuple(edges_sorted[edge][:, cell]))

    # Make the list of edges unique
    u_edge_list = np.unique(edge_list, axis=0)

    # Create a dictionary to store node to edge mapping
    n2e = dict()
    for idx, edge in enumerate(u_edge_list):
        n2e.update({tuple(edge): idx})
    ne = len(u_edge_list)

    # Populate cell_edges mapping matrix. This process is based on looping so that slow
    # performance should be expected. Basically, we loop through each edge (out of the six
    # possible edges) and then store the node number associated (from the node pair) to each
    # cell
    # Obtain edge-cell connectivity and store it in sparse form
    edge_cells_sparse = np.zeros((nc, ne), dtype=bool)
    edge_cells_dense = np.empty((2, nc, 6), dtype=np.int32)
    for edge in range(6):
        for node_pair, cell in zip(edges_sorted[edge].T, range(nc)):
            # Fill out mapping
            edge_cells_sparse[cell, n2e[tuple(node_pair)]] = True
            # Fill out dense mapping
            edge_cells_dense[0][cell][edge] = node_pair[0]
            edge_cells_dense[1][cell][edge] = node_pair[1]

    # Cell-edge mapping as a sparse matrix
    cell_edges_sparse = sps.csc_matrix(edge_cells_sparse).T

    # Obtain edge cardinality
    edge_cardinality = np.bincount(sps.find(cell_edges_sparse)[0])

    return cell_edges_sparse, edge_cells_dense


#%% Interpolation and polynomial-related functions
def interpolate_p1(point_val, point_coo):
    """
    Performs a linear local interpolation of a P1 FEM element given the
    the pressure values and the coordinates at the Lagrangian nodes.

    Parameters
    ----------
    point_val : NumPy nd-array of shape (g.num_cells x num_Lagr_nodes)
        Pressures values at the Lagrangian nodes.
    point_coo : NumPy nd-array of shape (g.dim x g.num_cells x num_Lagr_nodes)
        Coordinates of the Lagrangian nodes. In the case of embedded entities,
        the points should correspond to the rotated coordinates.

    Raises
    ------
    Value Error
        If the number of columns of point_val is different from 4 (3D), 3 (2d),
        or 2 (1D)

    Returns
    -------
    coeff : Numpy nd-array of shape (g.num_cells x (g.dim+1))
        Coefficients of the cell-wise P1 polynomial satisfying:
        c0 x + c1                   (1D),
        c0 x + c1 y + c2            (2D),
        c0 x + c1 y + c3 z + c4     (3D).
    """

    # Get rows, cols, and dimensionality
    rows = point_val.shape[0]  # number of cells
    cols = point_val.shape[1]  # number of Lagrangian nodes per cell
    if cols == 4:
        dim = 3
    elif cols == 3:
        dim = 2
    elif cols == 2:
        dim = 1
    else:
        raise ValueError("P1 reconstruction only valid for 1d, 2d, and 3d.")

    if dim == 3:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        z = point_coo[2].flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, y, z, ones])
        lcl = np.reshape(lcl, [rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, [rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    elif dim == 2:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, y, ones])
        lcl = np.reshape(lcl, [rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, [rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    else:
        x = point_coo.flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, ones])
        lcl = np.reshape(lcl, [rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, [rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    return coeff


def interpolate_p2(point_val, point_coo):
    """
    Performs a linear local interpolation of a P2 FEM element given the
    the pressure values and the coordinates at the Lagrangian nodes.

    Parameters
    ----------
    point_val : NumPy nd-array of shape (g.num_cells x dof)
        Pressures values at the Lagrangian nodes.
    point_coo : NumPy nd-array of shape (g.dim x g.num_cells x dof)
        Coordinates of the Lagrangian nodes. In the case of embedded entities,
        the points should correspond to the rotated coordinates.

    Raises
    ------
    Value Error
        If number of cols of point_val is different from 10 (3D), 6 (2D), or 3 (1D).

    Returns
    -------
    coeff : Numpy nd-array of shape (g.num_cells x dof)
        Coefficients of the cell-wise P2 polynomial satisfying:
        c0x^2 + c1x + c2                                                    (1D),
        c0x^2 + c1xy + c2x + c3y^2 + c4y + c5                               (2D),
        c0x^2 + c1xy + c2xz + c3x + c4y^2 + c5yz + c6y + c7z^2 + c8z + c9   (3D).
    """

    # Get rows, cols, and dimensionality
    rows = point_val.shape[0]  # number of cells
    cols = point_val.shape[1]  # number of Lagrangian nodes per cell
    if cols == 10:
        dim = 3
    elif cols == 6:
        dim = 2
    elif cols == 3:
        dim = 1
    else:
        raise ValueError("P2 reconstruction only valid for 1D, 2D, and 3D.")

    if dim == 3:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        z = point_coo[2].flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x ** 2, x * y, x * z, x, y ** 2, y * z, y, z ** 2, z, ones])
        lcl = np.reshape(lcl, [rows, 10, 10])

        p_vals = np.reshape(point_val, [rows, 10, 1])

        coeff = np.empty([rows, 10])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    elif dim == 2:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        ones = np.ones(6 * rows)

        lcl = np.column_stack([x ** 2, x * y, x, y ** 2, y, ones])
        lcl = np.reshape(lcl, [rows, 6, 6])

        p_vals = np.reshape(point_val, [rows, 6, 1])

        coeff = np.empty([rows, 6])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    else:
        x = point_coo.flatten()
        ones = np.ones(3 * rows)

        lcl = np.column_stack([x ** 2, x, ones])
        lcl = np.reshape(lcl, [rows, 3, 3])

        p_vals = np.reshape(point_val, [rows, 3, 1])

        coeff = np.empty([rows, 3])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    return coeff


def eval_p1(p1_coefficients: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Evaluate a P1 polynomial at the given coordinates

    Parameters
    ----------
    p1_coefficients (np.ndarray) : Polynomial to be evaluated, i.e., the one obtained from
        interpolate_P1. The expected shape is: rows x num_lagrangian_nodes.
    coordinates (np.ndarray): Coordinates with shape axes x rows x cols, where axes is
        the number of dimensions. If there is only one dimension present, we expect to
        to have 1 x row x cols nd-array.

    Raises
    ------
    ValueError: if there is any incosistencty in the shape of the inputs.

    Returns
    -------
    val (np.ndarray): Values of the polynomial at the coordinate points.
    """

    # Check if p1 coefficientes has the correct shape
    if p1_coefficients.shape[1] not in [2, 3, 4]:
        raise ValueError("Number of coefficients does not match a P1 polynomial")

    # Check if coordinates has the correct number of dimensions
    if len(coordinates.shape) != 3:
        raise ValueError("Coordinates array must be three-dimensional")

    # Retrieve coefficients
    c = poly2col(p1_coefficients)

    if len(c) == 4:
        val = c[0] * coordinates[0] + c[1] * coordinates[1] + c[2] * coordinates[2] + c[3]
    elif len(c) == 3:
        val = c[0] * coordinates[0] + c[1] * coordinates[1] + c[2]
    else:
        val = c[0] * coordinates[0] + c[1]

    return val


def eval_p2(p2_coefficients: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Evaluate a P2 polynomial at the given coordinates

    Parameters
    ----------
        p2_coefficients (np.ndarray) : Polynomial to be evaluated, i.e., the one obtained from
            interpolate_p2. The expected shape is g.num_cells x dof, where dof = 3 for 1D,
            dof = 6 for 2D, and dof = 10 for 3D.
        coordinates (np.ndarray): Coordinates with shape axes x rows x cols, where axes is
            the number of dimensions. If there is only one dimension present, we expect to
            to have 1 x row x cols nd-array.

    Raises
    ------
        ValueError: if there is any incosistencty in the shape of the inputs.

    Returns
    -------
        val (np.ndarray): Values of the polynomial at the coordinate points.
    """

    # Check if p1 coefficientes has the correct shape
    if p2_coefficients.shape[1] not in [3, 6, 10]:
        raise ValueError("Number of coefficients does not match a P2 polynomial")

    # Check if coordinates has the correct number of dimensions
    if len(coordinates.shape) != 3:
        raise ValueError("Coordinates array must be three-dimensional")

    # Retrieve coefficients
    c = poly2col(p2_coefficients)

    if len(c) == 10:  # 3D
        # c0x^2 + c1xy + c2xz + c3x + c4y^2 + c5yz + c6y + c7z^2 + c8z + c9
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        val = (
            c[0] * x ** 2
            + c[1] * x * y
            + c[2] * x * z
            + c[3] * x
            + c[4] * y ** 2
            + c[5] * y * z
            + c[6] * y
            + c[7] * z ** 2
            + c[8] * z
            + c[9]
        )
    elif len(c) == 6:  # 2D
        # c0x^2 + c1xy + c2x + c3y^2 + c4y + c5
        x = coordinates[0]
        y = coordinates[1]
        val = (
            c[0] * x ** 2
            + c[1] * x * y
            + c[2] * x
            + c[3] * y ** 2
            + c[4] * y
            + c[5]
        )
    else:  # 1D
        # c0x^2 + c1x + c2
        x = coordinates[0]
        val = (
            c[0] * x ** 2
            + c[1] * x
            + c[2]
        )

    return val


def poly2col(polynomial: np.ndarray) -> list:
    """
    Returns the coefficients (columns) of a polynomial in the form of a list.

    Parameters
    ----------
        polynomial (np.ndarray): Coefficients, i.e., the ones obtained from interpolate_P1. The
            expected shape is: rows x num_lagrangian_nodes.

    Returns
    -------
        List
            Coefficients stored in column-wise format.

    """
    rows = polynomial.shape[0]
    cols = polynomial.shape[1]
    coeff_list = []

    for col in range(cols):
        coeff_list.append(polynomial[:, col].reshape(rows, 1))

    return coeff_list
