import porepy as pp
import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps

#%% Geometry related function
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
        Rotated PorePy grid.
        
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
            dim,
            dim_bool,
            nodes,
        ):
    
            self.cell_centers = cell_centers
            self.face_normals = face_normals
            self.face_centers = face_centers
            self.rotation_matrix = rotation_matrix
            self.dim_bool = dim_bool
            self.dim = dim
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
                + "dim\n"
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
    dim = dim_bool.sum()
    rotated_object = RotatedGrid(
        cell_centers, face_normals, face_centers, rotation_matrix, dim, dim_bool, nodes
    )

    return rotated_object


def get_opposite_side_nodes(g):
    """
    Computes opposite side nodes for each face of each cell in the grid

    Parameters
    ----------
    g : PorePy object
        Grid

    Returns
    -------
    opposite_nodes : NumPy nd-array (g.num_cells x (g.dim + 1))
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
        Grid
    g_rot : Rotated grid object
        Rotated pseudo-grid 

    Returns
    -------
    sign_normals : NumPy nd-array of length g.num_faces
        Sign of the face normal. 1 if the signs of the local and global 
        normals are the same, -1 otherwise.
    """

    # We have to take care of the sign of the basis functions. The idea
    # is to create an array of signs "sign_normals" that will be multiplying
    # each edge basis function for the RT0 reconstruction of fluxes.
    # To determine this array, we need the following:
    #   (1) Compute the local outter normal (lon) vector for each cell
    #   (2) For every face of each cell, compare if lon == global normal vector.
    #       If they're not, then we need to flip the sign of lon for that face

    # Faces associated to each cell
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

    # Face centers coordinates for each face associated to each cell
    faceCntr_cells = g_rot.face_centers[:, faces_cell]

    # Global normals of the faces per cell
    glb_normal_faces_cell = g_rot.face_normals[:, faces_cell]

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


def get_quadpy_elements(g, g_rot):
    """
    Assembles the elements of a given grid in quadpy format
    For a 2D example see: https://pypi.org/project/quadpy/

    Parameters
    ----------
    g : PorePy object
        Grid
    g_rot: Rotated grid
        Rotated pseudo-grid
        
    Returns
    -------
    quadpy_elements : NumPy nd-array
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


def get_qp_elements_from_union_grid_1d(union_grid):
    
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
    
    # For some reason, quadpy needs a different formatting for line segments
    if dim == 1: 
        elements = elements.reshape(dim + 1, nc)
        
    return elements


#%% Interpolation and polynomial-related functions
def interpolate_P1(point_val, point_coo):
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
        c0x + c1                 (1D),
        c0x + c1y + c2           (2D),
        c0x + c1y + c3z + c4     (3D).
            
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
    

def eval_P1(P1_coeff, coor):
    """
    Evaluates a P1 polynomial at the given coordinates

    Parameters
    ----------
    P1 : NumPy nd-array of shape (rows, num_Lagr_nodes)
        Polynomial to be evaluated, i.e., the one obtained from interpolate_P1
    coor : NumPy nd-array of shape (dim, rows, num_Lagr_nodes)
        Coordinates with shape axes x rows x cols, where axes is the amount
        of dimensions. If there is only one dimension present, we expect to
        to have 1 xrows x cols nd-array.

    Raises
    ------
    ValueError
        If there is any incosistencty in the shape of the inputs.

    Returns
    -------
    val : NumPy nd-array
        Values of the polynomial at the coordinate points.

    """

    # Check if P1_coeff has the correct shape
    if P1_coeff.shape[1] not in [2, 3, 4]:
        raise ValueError("Number of coefficients does not match a P1 polynomial")
    
    # Check if coor has the correct number of dimensions
    if len(coor.shape) != 3:
        raise ValueError("Coordinates array must be three-dimensional")

    # Retrieve coefficients
    c = poly2col(P1_coeff)
    
    if len(c) == 4:
        val = c[0] * coor[0] + c[1] * coor[1] + c[2] * coor[2] + c[3]
    elif len(c) == 3:
        val = c[0] * coor[0] + c[1] * coor[1] + c[2]
    else:
        val = c[0] * coor[0] + c[1]
        
    return val
    
 
def poly2col(pol):
    """
    Returns the coefficients (columns) of a polynomial in the form of a list.

    Parameters
    ----------
    pol : NumPy nd-array of shape (rows, num_Lagr_nodes)
        Coefficients, i.e., the ones obtained from interpolate_P1 or interpolate_P2.

    Returns
    -------
    List
        Coefficients stored in column-wise format. 

    """
    rows = pol.shape[0]
    cols = pol.shape[1]
    coeff_list = []
    
    for col in range(cols):
        coeff_list.append(pol[:, col].reshape(rows, 1))

    return coeff_list

