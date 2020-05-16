import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import porepy as pp




def subdomain_pressure(gb, g, d, kw, sd_operator_name, p_name, lam_name, nodal_method, p_order):
    """
    Reconstructs subdomain pressures given the nodal method reconstruction and the 
    reconstruction order.

    Parameters
    ----------
    gb : PorePy object
        PorePy grid bucket object
    g : PorePy object
        Porepy grid object
    d : dictionary 
        Dictionary containing the parameters
    sd_operator_name    
    p_name : keyword
        Name of the subdomain variable
    lam_name : keyword
        Name of the edge variable
    nodal_method : keyword
        Name of the nodal reconstruction method: Either flux-inverse' or 'k-averaging'
    p_order : keyword
        Name of the pressure reconstruction order. Use '1' for P1 elements or '1.5' for P1 
        elements enriched with purely parabolic terms.

    Returns
    -------
    coeffs : NumPy array
        Coefficients of the reconstructed pressure for each cell.

    """

    # Compute nodal values of the pressure
    if nodal_method == "flux-inverse":
        p_nv = _compute_node_pressure_invflux(g, d, kw, sd_operator_name, p_name)
    elif nodal_method == "k-averaging":
        p_nv = _compute_node_pressure_kavg(g, d, kw, p_name)
    else:
        raise NameError("Nodal pressure reconstruction method not implemented")

    # Perform reconstruction for the given reconstruction order
    if p_order == "1":
        coeffs = _p1_reconstruction(g, p_nv)
    elif p_order == "1.5":
        p_cc = d[pp.STATE][p_name]
        coeffs = _p12_reconstruction(g, p_nv, p_cc)
    else:
        raise NameError("Pressure reconstruction order not implemented")

    
    d[pp.STATE]["p_coeff"] = coeffs
    
    return coeffs


def mono_grid_pressure(g, d, kw, p_name, nodal_method, p_order):
    """
    Reconstructs grid pressures given the nodal method reconstruction and the 
    reconstruction order.

    Parameters
    ----------
    g : PorePy object
        Porepy grid object
    d : dictionary 
        Dictionary containing the parameters
    p_name : keyword
        Name of the subdomain variable
    nodal_method : keyword
        Name of the nodal reconstruction method: Either flux-inverse' or 'k-averaging'
    p_order : keyword
        Name of the pressure reconstruction order. Use '1' for P1 elements or '1.5' for P1 
        elements enriched with purely parabolic terms.

    Returns
    -------
    coeffs : NumPy array
        Coefficients of the reconstructed pressure for each cell.

    """
    # Compute nodal values of the pressure
    if nodal_method == "flux-inverse":
        p_nv = _compute_node_pressure_invflux_mono(g, d, kw, p_name)
    elif nodal_method == "k_averaging":
        p_nv = _compute_node_pressure_kavg(g, d, kw, p_name)
    else:
        raise NameError("Nodal pressure reconstruction method not implemented")

    # Perform reconstruction for the given reconstruction order
    if p_order == "1":
        coeffs = _p1_reconstruction(g, p_nv)
    elif p_order == "1.5":
        p_cc = d[pp.STATE][p_name]
        coeffs = _p12_reconstruction(g, p_nv, p_cc)
    else:
        raise NameError("Pressure reconstruction order not implemented")

    d[pp.STATE]["p_coeff"] = coeffs

    return coeffs


def _compute_node_pressure_invflux(g, d, kw, sd_operator_name, p_name):
    """
    Computes nodal pressure values using the inverse of the flux

    Parameters
    ----------
    gb : PorePy object
        PorePy grid bucket object
    g : PorePy object
        Porepy grid object
    d : dictionary 
        Dicitionary containing the parameters
    kw : keyword
        Keyword referring to the problem type
    p_name : keyword
        Keyword for the subdomain variable
    lam_name : keyword
        Name of the edge variable

    Returns
    -------
    nodal_pressures : NumPy array
        Values of the pressure at the grid nodes.

    """

    # Retrieving topological data
    nc = g.num_cells
    nf = g.num_faces
    nn = g.num_nodes

    # Retrieve subdomain discretization
    discr = d[pp.DISCRETIZATION][p_name][sd_operator_name]

    # Boolean variable for checking is the scheme is FV
    is_fv = issubclass(type(discr), pp.FVElliptic)
    
    # Retrieve pressure from the dictionary
    if is_fv:
        p = d[pp.STATE][p_name]
    else:
        p = discr.extract_pressure(g, d[pp.STATE][p_name], d)

    # Perform reconstruction
    # NOTE: This is the original implementation of Eirik
    cell_nodes = g.cell_nodes()
    cell_node_volumes = cell_nodes * sps.dia_matrix((g.cell_volumes, 0), (nc, nc))
    sum_cell_nodes = cell_node_volumes * np.ones(nc)
    cell_nodes_scaled = (
        sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes
    )

    # Retrieving numerical fluxes
    if "full_flux" in d[pp.PARAMETERS][kw]:
        flux = d[pp.PARAMETERS][kw]["full_flux"]
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
    d[pp.STATE]["node_pressure"] = nodal_pressures

    return nodal_pressures


def _compute_node_pressure_invflux_mono(g, d, kw, p_name):
    """
    Computes nodal pressure values using the inverse of the flux

    Parameters
    ----------
    g : PorePy object
        Porepy grid object
    d : dictionary 
        Dicitionary containing the parameters
    kw : keyword
        Keyword referring to the problem type
    p_name : keyword
        Keyword for the subdomain variable

    Returns
    -------
    nodal_pressures : NumPy array
        Values of the pressure at the grid nodes.

    """

    # Retrieving topological data
    nc = g.num_cells
    nf = g.num_faces
    nn = g.num_nodes

    # Perform reconstruction
    # NOTE: This is the original implementation of Eirik
    cell_nodes = g.cell_nodes()
    cell_node_volumes = cell_nodes * sps.dia_matrix((g.cell_volumes, 0), (nc, nc))
    sum_cell_nodes = cell_node_volumes * np.ones(nc)
    cell_nodes_scaled = (
        sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes
    )

    # Retrieving numerical fluxes
    if "darcy_flux" in d[pp.PARAMETERS][kw]:
        flux = d[pp.PARAMETERS][kw]["darcy_flux"]
    else:
        raise('Darcy fluxes must be computed first')

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
            * (d[pp.STATE][p_name] + np.sum(dist * loc_grad, axis=0))
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
    d[pp.STATE]["node_pressure"] = nodal_pressures
    
    return nodal_pressures


def _compute_node_pressure_kavg(g, d, kw, p_name):
    """
    Computes nodal pressure values using k-averaging in a patch

    Parameters
    ----------
    g : PorePy object
        Porepy grid object
    d : dictionary 
        Dicitionary containing the parameters
    kw : string
        Keyword referring to the problem type
    p_name : string
        Keyword for the subdomain variable

    Returns
    -------
    nodal_pressures : NumPy array
        Values of the pressure at the grid nodes.

    """

    # Topological data
    nn = g.num_nodes
    nf = g.num_faces

    # Retrieve permeability values
    k = d[pp.PARAMETERS][kw]["second_order_tensor"].values
    # TODO: For the moment, we assume kxx = kyy = kzz on each cell
    # It would be nice to add the possibility to account for anisotropy
    k_broad = matlib.repmat(k[0][0], nn, 1)  # broaden array with number of nodes

    # Retrieve cell-centered pressures
    p_cc = d[pp.STATE][p_name]
    p_cc_broad = matlib.repmat(p_cc, nn, 1)  # broaden array with number of nodes

    # Create array of cell volumes. Note that the only the volumes of the cells
    # that belongs to each nodal patch are nonzero
    vol_broad = np.abs(g.cell_nodes().toarray() * matlib.repmat(g.cell_volumes, nn, 1))

    # Obtain nodal pressures applying the following formula on each node w:
    #
    #   pn_w = \sum_{i=1}^m (pc_i k_i |V_i|)/(k_i |V_i|),
    #
    # where i is the cell index, m is the number of cells sharing the common
    # node, pc_i is the cell-centered pressure, k_i is the cell permeability,
    # and |V_i| is the cell volume. Note that this
    numerator = np.sum(p_cc_broad * k_broad * vol_broad, axis=1)
    denoninator = np.sum(k_broad * vol_broad, axis=1)
    nodal_pressures = numerator / denoninator

    # Deal with Dirichlet and Neumann boundary conditions
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
    d[pp.STATE]["node_pressure"] = nodal_pressures

    return nodal_pressures


def _p1_reconstruction(g, p_nv):
    """
    Computes pressure reconstruction using P1 elements. The ouput is an array
    containing the coefficients needed for the linear reconstruction.

    For 1D:
        p(x) = a + bx
    For 2D:
        p(x,y) = a + bx + cy
    For 3D:
        p(x,y,z) = a + bx + cy + dz

    Parameters
    ----------
    g : PorePy object
        Porepy grid object
    p_nv : NumPy array
        Values of the pressure at the grid nodes.

    Returns
    -------
    coeffs : NumPy array (cell numbers x (g.dim + 1))
        Coefficients of the reconstructed pressure for each element of the grid

    """

    # TODO: Check what to do here to be consistent
    if g.dim == 0:
        return None

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


def _p12_reconstruction(g, p_nv, p_cc):
    """
    Computes pressure reconstruction using P1,2 elements. P1,2 elements are
    essentially P1 elements enriched with pure parabolic terms, i.e. x_i**2.
    The output is an array containing the coefficients of the reconstruction.

    For 1D:
        p(x) = a + bx + ex^2
    For 2D:
        p(x,y) = a + bx + cy + e(x^2 + y^2)
    For 3D:
        p(x,y,z) = a + bx + cy + dz + e(x^2 + y^2 + z^2)
        
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
    coeffs : NumPy array (cell numbers x (2*g.dim + 1))
        Coefficients of the reconstructed pressure for each element of the grid    
    
    """

    # TODO: Check what to do here to be consistent
    if g.dim == 0:
        return None

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
        parabola_terms += x ** 2
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
