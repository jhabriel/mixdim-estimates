# Importing modules
import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import porepy as pp

from porepy.grids.grid_bucket import GridBucket


def subdomain_velocity(gb, g, g_rot, d, kw, lam_name):
    """
    Computes mixed-dimensional flux reconstruction using RT0 extension of 
    normal full fluxes

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
    coeffs : NumPy array (cell_numbers x (g.dim+1))
        Coefficients of the reconstructed velocity for each element.
        
    cc_vel: Numpy array (cell numbers x g.dim)
        Components of the reconstructed velocities evaluated at the cell centers.
    
    """

    # Mappings
    cell_faces_map, _, _ = sps.find(g_rot.cell_faces)
    cell_nodes_map, _, _ = sps.find(g_rot.cell_nodes())

    # Cell-wise arrays
    faces_cell = cell_faces_map.reshape(np.array([g_rot.num_cells, g.dim + 1]))
    nodes_cell = cell_nodes_map.reshape(np.array([g_rot.num_cells, g.dim + 1]))
    opp_nodes_cell = _get_opposite_side_nodes(g_rot)
    sign_normals_cell = _get_sign_normals(g_rot)
    vol_cell = g_rot.cell_volumes

    # Opposite side nodes for RT0 extension of normal fluxes
    opp_nodes_coor_cell = np.empty(
        [g_rot.dim, nodes_cell.shape[0], nodes_cell.shape[1]]
    )
    for dim in range(g.dim):
        opp_nodes_coor_cell[dim] = g_rot.nodes[dim][opp_nodes_cell]

    # Obtain the mixed-dimensional full flux. That is, the darcy flux + the
    # mortar projection of the lower-dimensional neighboring interfaces
    if "full_flux" in d[pp.PARAMETERS][kw]:
        full_flux = d[pp.PARAMETERS][kw]["full_flux"]
    else:
        raise ("Full flux must be computed first")

    # Local full flux divergence
    full_flux_local_div = (sign_normals_cell * full_flux[faces_cell]).sum(axis=1)

    # Check if mass conservation is satisfied on a cell basis, in order to do
    # this, we check on a local basis, if the divergence of the flux equals
    # the sum of internal and external source terms
    ext_src = d[pp.PARAMETERS][kw]["source"]
    int_src = _internal_source_term_contribution(gb, g, lam_name)
    np.testing.assert_almost_equal(
        full_flux_local_div,
        ext_src + int_src,
        decimal=7,
        err_msg="Error estimates only implemented for local mass-conservative methods",
    )

    # Determining coefficients
    coeffs = np.empty([g_rot.num_cells, g_rot.dim + 1])
    alpha = 1 / (g.dim * vol_cell)
    coeffs[:, 0] = alpha * np.sum(sign_normals_cell * full_flux[faces_cell], axis=1)
    for dim in range(g_rot.dim):
        coeffs[:, dim + 1] = -alpha * np.sum(
            (sign_normals_cell * full_flux[faces_cell] * opp_nodes_coor_cell[dim]),
            axis=1,
        )

    # Check if the reconstructed evaluated at the face centers normal fluxes
    # match the numerical ones
    recons_flux = _get_reconstructed_face_fluxes(g_rot, coeffs)
    np.testing.assert_almost_equal(
        recons_flux, full_flux, decimal=8, err_msg="Flux reconstruction has failed"
    )

    return coeffs


def interface_fluxes(d_e):
    
    g_m = d_e['mortar_grid']    
    
    return None



def mono_grid_velocity(g, d, kw, p_name, sd_operator_name):
    """
    Computes flux reconstruction in the case of mono-dimensional grids

    Parameters
    ----------
    g : PorePy object
        PorePy grid object
    d : Dictionary
        Data dictionary
    kw : Keyword
        Name of the problem, e.g., flow
    p_name: Keyword
        Grid variable, e.g., pressure
    sd_operator_name : Keyword
        Grid operator name, e.g., diffusion

    Returns
    -------
    coeffs : NumPy array (cell_numbers x (g.dim+1))
        Coefficients of the reconstructed velocity for each element.
        
    cc_vel: Numpy array (cell numbers x g.dim)
        Components of the reconstructed velocities evaluated at the cell centers.

    """

    # Retrieve subdomain discretization
    discr = d["discretization"][p_name][sd_operator_name]

    # Check if the scheme if is fv
    is_fv = issubclass(type(discr), pp.FVElliptic)

    # Mappings
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())

    # Cell-wise arrays
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    opp_nodes_cell = _get_opposite_side_nodes(g)
    sign_normals_cell = _get_sign_normals(g)
    vol_cell = g.cell_volumes

    # Opposite side nodes
    opp_nodes_coor_cell = np.empty([g.dim, nodes_cell.shape[0], nodes_cell.shape[1]])
    for dim in range(g.dim):
        opp_nodes_coor_cell[dim] = g.nodes[dim][opp_nodes_cell]

    # Retrieving darcy fluxes
    if is_fv:
        if "darcy_flux" in d[pp.PARAMETERS][kw]:
            darcy_flux = d[pp.PARAMETERS][kw]["darcy_flux"]
        else:
            pp.fvutils.compute_darcy_flux(g, data=d)
    else:
        darcy_flux = discr.extract_flux(g, d[pp.STATE][p_name], d)

    # Fluxes for each cell
    flux_cell = sign_normals_cell * darcy_flux[faces_cell]

    # Check if mass conservation is satisfied on a cell basis
    src = d[pp.PARAMETERS][kw]["source"]
    np.testing.assert_almost_equal(
        flux_cell.sum(axis=1),
        src,
        decimal=7,
        err_msg="Error estimates only implemented for local mass-conservative methods",
    )

    # Determining coefficients
    coeffs = np.empty([g.num_cells, g.dim + 1])
    alpha = 1 / (g.dim * vol_cell)
    coeffs[:, 0] = alpha * np.sum(flux_cell, axis=1)
    for dim in range(g.dim):
        coeffs[:, dim + 1] = -alpha * np.sum(
            (flux_cell * opp_nodes_coor_cell[dim]), axis=1
        )

    # Check if the reconstructed normal velocities evaluated at the face centers
    # match the numerical ones
    recons_flux = _get_reconstructed_face_fluxes(g, coeffs)
    np.testing.assert_almost_equal(
        recons_flux, darcy_flux, decimal=8, err_msg="Flux reconstruction has failed"
    )

    # Evaluating velocities at the cell centers
    # NOTE: This is not necessary for the purpose of evaluating the a posteriori
    # errors, but it might be useful for others
    coords = g.cell_centers.transpose()
    cc_vel = np.zeros([g.num_cells, g.dim])
    for dim in range(g.dim):
        cc_vel[:, dim] = coeffs[:, 0] * coords[:, dim] + coeffs[:, dim + 1]

    return coeffs, cc_vel


def compute_full_flux_fv(
    gb, keyword, sd_operator_name, p_name, lam_name,
):
    """
    Computes full flux over all faces for all the subdomains of the grid bucket.
    The full flux is composed by the Darcy flux plus the projection of the 
    lower-dimensional neighboring mortar fluxes. This function should be used
    for finite-volume discretization schemes

    Parameter:
    gb: grid bucket with the following data fields for all nodes/grids:
        'flux': Internal discretization of fluxes.
        'bound_flux': Discretization of boundary fluxes.
        'pressure': Pressure values for each cell of the grid (overwritten by p_name).
        'bc_val': Boundary condition values.
            and the following edge property field for all connected grids:
        'coupling_flux': Discretization of the coupling fluxes.
    keyword (str): defaults to 'flow'. The parameter keyword used to obtain the
        data necessary to compute the fluxes.
    keyword_store (str): defaults to keyword. The parameter keyword determining
        where the data will be stored.
    d_name (str): defaults to 'darcy_flux'. The parameter name which the computed
        darcy_flux will be stored by in the dictionary.
    p_name (str): defaults to 'pressure'. The keyword that the pressure
        field is stored by in the dictionary.
    lam_name (str): defaults to 'mortar_solution'. The keyword that the mortar flux
        field is stored by in the dictionary.

    Returns:
        gb, the same grid bucket with the added field 'full_flux' added to all
        node data fields. 
    """

    # Compute fluxes from pressures internal to the subdomain, and for global
    # boundary conditions.
    for g, d in gb:

        # Compute Darcy flux
        if g.dim > 0:
            parameter_dictionary = d[pp.PARAMETERS][keyword]
            matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][keyword]
            darcy_flux = (
                matrix_dictionary["flux"] * d[pp.STATE][p_name]
                + matrix_dictionary["bound_flux"] * parameter_dictionary["bc_values"]
            )

            d[pp.PARAMETERS][keyword]["full_flux"] = darcy_flux

    # Add the contribution of the mortar fluxes for each edge associated
    # to a given subdomain
    for e, d in gb.edges():

        g_m = d["mortar_grid"]
        g_h = gb.nodes_of_edge(e)[1]
        d_h = gb.node_props(g_h)

        # The mapping mortar_to_hat_bc contains is composed of a mapping to
        # faces on the higher-dimensional grid, and computation of the induced
        # fluxes.
        bound_flux = d_h[pp.DISCRETIZATION_MATRICES][keyword]["bound_flux"]
        induced_flux = bound_flux * g_m.mortar_to_master_int() * d[pp.STATE][lam_name]

        d_h[pp.PARAMETERS][keyword]["full_flux"] += induced_flux


def compute_full_flux_rt0(
    gb, keyword, sd_operator_name, p_name, lam_name,
):
    """
    Computes full flux over all faces for all the subdomains of the grid bucket.
    The full flux is composed by the Darcy flux plus the projection of the 
    lower-dimensional neighboring mortar fluxes. This function should be used
    for finite-element discretization schemes

    Parameter:
    gb: grid bucket with the following data fields for all nodes/grids:
        'flux': Internal discretization of fluxes.
        'bound_flux': Discretization of boundary fluxes.
        'pressure': Pressure values for each cell of the grid (overwritten by p_name).
        'bc_val': Boundary condition values.
            and the following edge property field for all connected grids:
        'coupling_flux': Discretization of the coupling fluxes.
    keyword (str): defaults to 'flow'. The parameter keyword used to obtain the
        data necessary to compute the fluxes.
    keyword_store (str): defaults to keyword. The parameter keyword determining
        where the data will be stored.
    d_name (str): defaults to 'darcy_flux'. The parameter name which the computed
        darcy_flux will be stored by in the dictionary.
    p_name (str): defaults to 'pressure'. The keyword that the pressure
        field is stored by in the dictionary.
    lam_name (str): defaults to 'mortar_solution'. The keyword that the mortar flux
        field is stored by in the dictionary.

    Returns:
        gb, the same grid bucket with the added field 'full_flux' added to all
        node data fields. 
    """

    # Loop through all the subdomains
    for g, d in gb:

        # Retrieve subdomain discretization
        discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

        # Retrieve darcy flux from the solution array
        darcy_flux = discr.extract_flux(g, d[pp.STATE][p_name], d)

        # we need to recover the flux from the mortar variable before
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
                    sign * g_m.master_to_mortar_avg().T * d_e[pp.STATE][lam_name]
                )

        d[pp.PARAMETERS][keyword]["full_flux"] = darcy_flux + induced_flux


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
    
    The data dicitionary of each subdomain is updated with the field 
    d[pp.PARAMETERS][kw]["full_flux"], which is an NumPy array of length
    g.num_faces.

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
                d[pp.PARAMETERS][kw]["full_flux"] = darcy_flux + induced_flux

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
                d[pp.PARAMETERS][kw]["full_flux"] = darcy_flux + induced_flux

    return


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
    int_source : NumPy array (g.num_cells)
        Flux contribution from higher-dimensional neighboring interfaces to the
        lower-dimensional grid g, in the form of source term

    """

    # Initialize internal source term
    int_source = np.zeros(g.num_cells)

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
        int_source = g_mortar.mortar_to_slave_int() * mortar_flux

    return int_source


def _get_mortar_flux_contribution(gb, g):
    """
    Obtain mortar flux contribution from lower-dimensional neighboring interfaces
    
    Parameters
    ----------
    gb : PorePy object
        PorePy grid bucket object.
    grid : PorePy object
        Porepy grid object.

    Returns
    -------
    mortar_contribution : NumPy array (g.num_faces)
        Flux contribution from the interfaces to the higher-dimensional grid g.

    """

    # Initialize mortar contribution array
    mortar_contribution = np.zeros(g.num_faces)

    # Obtain lower dimensional neighboring nodes
    g_lows = gb.node_neighbors(g, only_lower=True)

    # We loop through all the lower dimensional adjacent interfaces to
    # the higher-dimensional map the mortar fluxes to the higher-dimensional
    # subdomain
    for g_low in g_lows:

        # Retrieve the dictionary and mortar grid of the corresponding edge
        d_edge = gb.edge_props([g_low, g])
        g_mortar = d_edge["mortar_grid"]

        # Obtain faces of the higher-dimesnional grid to which the mortar fluxes
        # will be mapped to
        mortar_to_master_faces, _, _ = sps.find(g_mortar.mortar_to_master_int())

        # Retrieve mortar fluxes
        mortar_flux = d_edge[pp.STATE]["interface_flux"]

        # Obtain mortar flux contribution associated to the neighboring interface
        mortar_contribution[mortar_to_master_faces] += mortar_flux

    return mortar_contribution


# -------------------------------------------------------------------------- #
#                      Assert/Testing related functions                      #
# -------------------------------------------------------------------------- #
def _get_reconstructed_face_fluxes(g, coeff):
    """
    Obtain reconstructed fluxes at the cell centers for a given mesh

    Parameters
    ----------
    g : PorePy object
        PorePy grid object. Note that for mixed-dimensional grids, this will
        correspond to the rotated grid object.
    coeff : TYPE
        DESCRIPTION.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

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
    out = Q_flat[idx_q]

    return out


# -------------------------------------------------------------------------- #
#                           Utility functions                                #
# -------------------------------------------------------------------------- #
def _get_opposite_side_nodes(g):
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


def _get_sign_normals(g):
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
