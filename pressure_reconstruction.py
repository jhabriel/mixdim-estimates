# Importing modules
import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import porepy as pp

# Compute pressure reconstruction
def subdomain_pressure(grid, data, parameter_keyword, subdomain_variable, nodal_method, p_recons_order):
    """
    Reconstructs subdomain pressures given the nodal method reconstruction and the 
    reconstruction order. By default, it is assumed a conforming P1-reconstruction
    from nodal values obtained using a permeability-weighted average.

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object
    data : dictionary 
        Dicitionary containing the parameters
    subdomain_variable : string
        Keyword for the subdomain variable
    nodal_method : string
        Nodal reconstruction method: Either 'k-averaging' or 'mpfa-inverse'
    p_recons_order : string
        Pressure reconstruction order. Use '1' for P1 elements or '1.5' for P1 
        elements enriched with purely parabolic terms.

    Returns
    -------
    coeffs : NumPy array
        Coefficients of the reconstructed pressure for each element.

    """
       
    # Compute nodal values of the pressure
    if nodal_method == "k-averaging":
        p_nv = _compute_node_pressure_kav(grid, data, parameter_keyword, subdomain_variable)
    elif nodal_method == "mpfa-inverse":
        p_nv = _compute_node_pressure_inv(grid, data, parameter_keyword, subdomain_variable)
    else:
        raise NameError('Nodal reconstruction method not implemented')

    # Perform reconstruction for the given reconstruction order
    if p_recons_order == "1":
        coeffs = _p1_reconstruction(grid, p_nv)
    elif p_recons_order == "1.5":
        p_cc = data[pp.STATE][subdomain_variable]
        coeffs = _p12_reconstruction(grid, p_nv, p_cc)
    else:
        raise NameError('Pressure reconstruction order not implemented')

    return coeffs

#%% Compute nodal values using inverse MPFA
def _compute_node_pressure_inv(grid, data, parameter_keyword, subdomain_variable):
    """
    Computes nodal pressure values using the inverse mpfa technique

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object
    data : dictionary 
        Dicitionary containing the parameters
    parameter_keyword : string
        Keyword referring to the problem type
    subdomain_variable : string
        Keyword for the subdomain variable

    Returns
    -------
    nodal_pressures : NumPy array
        Values of the pressure at the grid nodes.

    """
    
    # Renaming variables
    g = grid
    d = data
    kw_f = parameter_keyword
    sd_var = subdomain_variable
    
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
    if "darcy_flux" in d[pp.PARAMETERS][kw_f]:
        flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]
    else:
        pp.fvutils.compute_darcy_flux(g, keyword=kw_f, data=d)
        flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]      
    proj_flux = pp.RT0(kw_f).project_flux(g, flux, d)[: g.dim]

    loc_grad = np.zeros((g.dim, nc))
    perm = d[pp.PARAMETERS][kw_f]["second_order_tensor"].values

    for ci in range(nc):
        loc_grad[: g.dim, ci] = -np.linalg.inv(perm[: g.dim, : g.dim, ci]).dot(
            proj_flux[:, ci]
        )

    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    cell_node_matrix = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
    nodal_pressures = np.zeros(nn)

    for col in range(g.dim + 1):
        nodes = cell_node_matrix[:, col]
        dist = g.nodes[: g.dim, nodes] - g.cell_centers[: g.dim]
        scaling = cell_nodes_scaled[nodes, np.arange(nc)]
        contribution = (
            np.asarray(scaling)
            * (d[pp.STATE][sd_var] + np.sum(dist * loc_grad, axis=0))
        ).ravel()
        nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)

    bc = d[pp.PARAMETERS][kw_f]["bc"]
    bc_values = d[pp.PARAMETERS][kw_f]["bc_values"]

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

    return nodal_pressures


#%% Compute nodal values using k-averaging
def _compute_node_pressure_kav(grid, data, parameter_keyword, subdomain_variable):
    """
    Computes nodal pressure values using k-averaging in a patch

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object
    data : dictionary 
        Dicitionary containing the parameters
    parameter_keyword : string
        Keyword referring to the problem type
    subdomain_variable : string
        Keyword for the subdomain variable

    Returns
    -------
    nodal_pressures : NumPy array
        Values of the pressure at the grid nodes.


    """
    
    #TODO: This does not take into account the values of the permeabilities
    #This must be implemented ASAP
    
    # Renaming variables
    g = grid
    d = data
    kw_f = parameter_keyword
    sd_var = subdomain_variable

    # Topological data
    nn = g.num_nodes
    nf = g.num_faces

    # Performing reconstuction
    p_cc = d[pp.STATE][sd_var]
    p_cc_broad = matlib.repmat(p_cc, nn, 1)
    vol_broad = g.cell_nodes().toarray() * matlib.repmat(g.cell_volumes, nn, 1)
    num = np.sum(p_cc_broad * vol_broad, axis=1)
    den = np.sum(vol_broad, axis=1)
    nodal_pressures = num / den

    bc = d[pp.PARAMETERS][kw_f]["bc"]
    bc_values = d[pp.PARAMETERS][kw_f]["bc_values"]
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

    return nodal_pressures


#%% Compute P1 pressure reconstruction
def _p1_reconstruction(grid, nodal_pressures):
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
    grid : PorePy object
        Porepy grid object
    nodal_pressures : NumPy array
        Values of the pressure at the grid nodes.

    Returns
    -------
    coeffs : NumPy array (cell numbers x (g.dim + 1))
        Coefficients of the reconstructed pressure for each element of the grid

    """
 
    # Renaming variables
    g = grid
    p_nv = nodal_pressures
       
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
    vertices_pressures = p_nv[nodes_cell] # pressure at the vertices
    for cell in range(g.num_cells):
        inv_local_matrix = np.linalg.inv(lcl[cell])
        vert_press_cell = vertices_pressures[cell]
        coeffs[cell] = np.dot(inv_local_matrix, vert_press_cell)

    return coeffs


#%% Compute P1 pressure reconstruction enriched with purely parabolic terms
def _p12_reconstruction(grid, nodal_pressures, cell_center_pressures):
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
    grid : PorePy object
        Porepy grid object
    nodal_pressures : NumPy array
        Values of the pressure at the grid nodes.
    cell_center_pressures : Numpy array
        Values of the pressure at the cell centers.

    Returns
    -------
    coeffs : NumPy array (cell numbers x (2*g.dim + 1))
        Coefficients of the reconstructed pressure for each element of the grid    
    """

    # Renaming variables
    g = grid
    p_nv = nodal_pressures
    p_cc = cell_center_pressures
    
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
