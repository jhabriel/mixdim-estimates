# Importing modules
import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import porepy as pp


def subdomain_velocity(grid, data, parameter_keyword):
    """
    Computes flux reconstruction using RT0 extension of normal fluxes

    The output array contains the coefficients satisfying the
    following velocity fields depending on the dimensionality of the problem:
        
    q = ax + b                          (for 1d),
    q = (ax + b, ay + c)^T              (for 2d),
    q = (ax + b, ay + c, az + d)^T      (for 3d).
    
    For an element, the reconstructed velocity field inside an element K
    is given by:
    
        q = \sum_{j=1}^{g.dim+1} q_j psi_j, 
        
    where psi_j are the global basis functions defined on each face, 
    and q_j are the normal fluxes.

    The global basis takes the form
        
    psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i)^T                      (for 1d),
    psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i)^T             (for 2d),
    psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i, z - z_i)^T    (for 3d),
    
    where s(normal_j) is the sign of the normal vector,|K| is the volume 
    of the element, and (x_i, y_i, z_i) are the coordinates of the 
    opposite side nodes to the face j.

    The funcion s(normal_j) = 1 if the signs of the local and global 
    normals are the same, and -1 otherwise.

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object.
    data : dictionary 
        Dicitionary containing the parameters.
    parameter_keyword : string
        Keyword referring to the problem type.

    Returns
    ------
    coeffs : NumPy array (cell_numbers x (g.dim+1))
        Coefficients of the reconstructed velocity for each element.
        
    cc_vel: Numpy array (cell numbers x g.dim)
        Components of the reconstructed velocities evaluated at the cell centers.
    
    """
   
    # Renaming variables 
    g = grid
    d = data
    kw = parameter_keyword
    coords = g.cell_centers.transpose() # cell centers coordinates
    
    # Mappings
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())

    # Cell-wise arrays
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    opp_nodes_cell = _get_opposite_side_nodes(g)
    sign_normals_cell = _get_sign_normals(g)
    vol_cell = g.cell_volumes

    opp_nodes_coor_cell = np.empty([g.dim, nodes_cell.shape[0], nodes_cell.shape[1]])
    for dim in range(g.dim):
        opp_nodes_coor_cell[dim] = g.nodes[dim][opp_nodes_cell]

    # Retrieving numerical fluxes
    if "darcy_flux" in d[pp.PARAMETERS][kw]:
        fluxes = d[pp.PARAMETERS][kw]["darcy_flux"]
    else:
        pp.fvutils.compute_darcy_flux(g, keyword=kw, data=d)
        fluxes = d[pp.PARAMETERS][kw]["darcy_flux"]
    fluxes_cell = fluxes[faces_cell]

    # Determining coefficients
    coeffs = np.empty([g.num_cells, g.dim + 1])
    alpha = 1 / (g.dim * vol_cell)
    coeffs[:, 0] = alpha * np.sum(sign_normals_cell * fluxes_cell, axis=1)
    for dim in range(g.dim):
        coeffs[:, dim + 1] = -alpha * np.sum(
            (sign_normals_cell * fluxes_cell * opp_nodes_coor_cell[dim]), axis=1
        )

    # Evaluating velocities at the cell centers
    # NOTE: This is not necessary for the purpose of evaluating the a posteriori
    # errors, but it might be useful for others
    cc_vel = np.zeros([g.num_cells, g.dim])
    for dim in range(g.dim):
        cc_vel[:, dim] = coeffs[:, 0] * coords[:, dim] + coeffs[:, dim+1]

    return coeffs, cc_vel


def _get_opposite_side_nodes(grid):
    """
    Computes opposite side nodes for each face of each cell in the grid

    The ouput is a two-dimensional array, 

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object.

    Returns
    ------
    opposite_nodes : NumPy array (cell_numbers x (g.dim + 1))
        Array where the rows represent the cell number, and the columns 
        represent the opposite side node index of the face.
    """
    
    
    # Rename variable
    g = grid

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


def _get_sign_normals(grid):
    """
    Computes the sign of the face normals for each element in the grid

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object.

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

    # Rename variable
    g = grid
    
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