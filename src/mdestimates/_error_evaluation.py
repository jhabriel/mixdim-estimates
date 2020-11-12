import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import quadpy as qp
import porepy as pp

import mdestimates.estimates_utils as utils

#%% Error computation
def compute_error_estimates(self):
    """
    Computes error estimates for all nodes and edges of the Grid Bucket.

    Returns
    -------
    None.

    """

    # Loop through all the nodes of the grid bucket
    for g, d in self.gb:

        # Handle the case of zero-dimensional subdomains
        if g.dim == 0:
            d[self.estimates_kw]["diffusive_error"] = None
            continue

        # Rotate grid. If g == gb.dim_max(), this has no effect whatsoever
        g_rot = utils.rotate_embedded_grid(g)

        # Obtain the diffusive flux error
        diffusive_error = subdomain_diffusive_error(self, g, g_rot, d)
        d[self.estimates_kw]["diffusive_error"] = diffusive_error

    # Loop through all the edges of the grid bucket
    for e, d_e in self.gb.edges():

        # Obtain the diffusive flux error
        diffusive_error = interface_diffusive_error(self, e, d_e)
        d_e[self.estimates_kw]["diffusive_error"] = diffusive_error

    return None


#%% Subdomain error
def subdomain_diffusive_error(self, g, g_rot, d):
    """
    Computes the (square) of the subdomain diffusive errors, given (locally) for
    an element E by:
        
                || K^(-1/2) u^RT0_E + K^(1/2) grad(p_E) ||_E^2,
        
    where K the tangential permeability, u^RT0 is the RT0 extended velocity field, 
    and grad(p) is the gradient of the reconstructed pressure.

    Parameters
    ----------
    g : PorePy Object
        PorePy grid object
    g_rot : Rotated Object
        Rotated pseudo-grid
    d : Dictionary
        Data dictionary.

    Raises
    ------
    ValueError
        If the pressure has not been reconstructed
        If the velocity has not been reconstructed
        If the grid dimension is not 1D, 2D, or 3D

    Returns
    -------
    diffusive_error : NumPy nd-array of size (g.num_cells)
        Diffusive flux error estimates (squared) for the given grid
    
    """

    # Run sanity checks
    if g.dim not in [1, 2, 3]:
        raise ValueError("Error not defined for the given grid dimension")
    if "recon_p" not in d[self.estimates_kw]:
        raise ValueError("Pressure must be reconstructed first")
    if "recon_u" not in d[self.estimates_kw]:
        raise ValueError("Velocity must be reconstructed first")

    # Retrieve postprocessed pressure
    recon_p = d[self.estimates_kw]["recon_p"].copy()
    
    # Retrieve reconstructed velocity
    recon_u = d[self.estimates_kw]["recon_u"].copy()
    
    # Retrieve permeability
    perm = d[pp.PARAMETERS][self.kw]["second_order_tensor"].values
    k = perm[0][0].reshape(g.num_cells, 1)

    # Get quadpy elements and declare integration method
    #TODO: Use the correct degree of quadrature according to p_degree
    elements = utils.get_quadpy_elements(g, g_rot)
    if g.dim == 1:
        method = qp.c1.newton_cotes_closed(3)
    elif g.dim == 2:
        method = qp.t2.get_good_scheme(3)
    else:
        method = qp.t3.get_good_scheme(3)

    # Obtain coefficients 
    p = utils.poly2col(recon_p)
    u = utils.poly2col(recon_u)

    # Declare integrands and prepare for integration
    def integrand(x):
        
        if g.dim == 1:
            
            veloc_x = u[0] * x + u[1]
            
            gradp_x = p[0] * np.ones_like(x)
            
            int_x = (k ** (-0.5) * veloc_x + k ** (0.5) * gradp_x) ** 2
            
            return int_x
        
        elif g.dim == 2:
            
            veloc_x = u[0] * x[0] + u[1]
            veloc_y = u[0] * x[1] + u[2]
            
            gradp_x = p[0] * np.ones_like(x[0])
            gradp_y = p[1] * np.ones_like(x[1])
            
            int_x = (k ** (-0.5) * veloc_x + k ** (0.5) * gradp_x) ** 2
            int_y = (k ** (-0.5) * veloc_y + k ** (0.5) * gradp_y) ** 2
            
            return int_x + int_y
        
        else:
            
            veloc_x = u[0] * x[0] + u[1]
            veloc_y = u[0] * x[1] + u[2]
            veloc_z = u[0] * x[2] + u[3]
            
            gradp_x = p[0] * np.ones_like(x[0])
            gradp_y = p[1] * np.ones_like(x[1])
            gradp_z = p[2] * np.ones_like(x[2])
            
            int_x = (k ** (-0.5) * veloc_x + k ** (0.5) * gradp_x) ** 2
            int_y = (k ** (-0.5) * veloc_y + k ** (0.5) * gradp_y) ** 2
            int_z = (k ** (-0.5) * veloc_z + k ** (0.5) * gradp_z) ** 2
            
            return int_x + int_y + int_z

    # Compute the integral
    diffusive_error = method.integrate(integrand, elements)

    return diffusive_error


#%% Interface error
def interface_diffusive_error(self, e, d_e):
    """
    Wrapper for obtaining the diffusive error on the interfaces given (locally) 
    for a mortar element E by:
        
            || k^(-1/2) \lambda_E + k^(1/2) (pl_E - tr ph_E) ||_E^2,

    where k is the normal diffusivity, \lambda the normal velocity 
    (scaled mortar flux), pl is the lower-dimensional pressure, and tr ph 
    is the trace of the higher-dimensional pressure.

    Parameters
    ----------
    e : PorePy object
        Edge
    d_e : Dictionary
        Edge data dictionary.

    Raises
    ------
    ValueError
        If the mortar grid dimension is not 0D, 1D, or 2D.

    Returns
    -------
    diffusive_error : NumPy nd-array
        Diffusive error (squared) for each element of the mortar grid

    """
    
    # Get hold of mortar grid
    mg = d_e["mortar_grid"]
    
    # Check dimensionality of the mortar grid
    if mg.dim not in [0, 1, 2]:
        raise ValueError('Inconsistent mortar grid dimension. Expected 0, 1, or 2.')
    
    # Retrieve connectivity maps for checking if we have non-matching grid
    primary2mortar_hits = mg.primary_to_mortar_avg().nonzero()[0].size
    secondary2mortar_hits = mg.secondary_to_mortar_avg().nonzero()[0].size
    nc = mg.num_cells
    
    
    # Obtain diffusive error depending on the dimensionality of the grid
    if mg.dim == 0:
        diffusive_error = _interface_diffusive_error_0d(self, e, d_e)
    elif mg.dim == 1:
        if (primary2mortar_hits == nc) and (secondary2mortar_hits == nc):    
            diffusive_error = _interface_diffusive_error_1d(self, e, d_e)
        else:
            diffusive_error = _interface_diffusive_error_nonmatching_1d(self, e, d_e)
    else:
        if (primary2mortar_hits == nc) and (secondary2mortar_hits == nc):    
            diffusive_error = _interface_diffusive_error_2d(self, e, d_e)
        else:
            raise ValueError('Non-matching grids in 3D are not implemented')
        
    return diffusive_error


def _get_high_pressure_trace(self, g_l, g_h, d_h, frac_faces):
    """
    Obtains the coefficients of the P1 (projected) traces of the pressure
    
    Parameters
    ----------
    g_l : PorePy object
        Lower-dimensional grid.
    g_h : PorePy object
        Higher-dimensional grid.
    d_h : Dictionary
        Higher-dimensional data dictionary.
    frac_faces : NumPy nd-array
        Higher-dimensional fracture faces

    Raises
    ------
    ValueError
        If the pressure has not been reconstructed

    Returns
    -------
    trace_pressure : NumPy nd-array
        Coefficients of the trace of the pressure

    """
    
    def get_edge_lag_coo(self, grid):
        """
        Utility function to get hold of the coordinates of the Lagrangian nodes 
        corresponding to the internal higher-dimensional boundary 

        Parameters
        ----------
        grid : PorePy grid
            Higher-dimensional grid.

        Returns
        -------
        lagran_coo : NumPy nd-array
            Coordinates of the Lagrangian nodes.

        """
        # Get nodes of the fracture fraces
        nodes_of_frac_faces = np.reshape(
            sps.find(g_h.face_nodes.T[frac_faces].T)[0], 
            [frac_faces.size, g_h.dim]
            )
        
        # Obtain the coordinates of the nodes of the fracture faces
        lagran_coo = grid.nodes[:, nodes_of_frac_faces]
        
        return lagran_coo
       
    # Rotate both grids, and obtain rotation matrix and effective dimension
    gh_rot = utils.rotate_embedded_grid(g_h)
    gl_rot = utils.rotate_embedded_grid(g_l)
    R = gl_rot.rotation_matrix
    dim_bool = gl_rot.dim_bool
    
    # Obtain the cells coorresponding to the frac_faces
    cells_of_frac_faces, _, _ = sps.find(g_h.cell_faces[frac_faces].T)      
    
    # Retrieve the coefficients of the polynomials corresponding to those cells
    if "recon_p" in d_h[self.estimates_kw]:
        p_high = d_h[self.estimates_kw]["recon_p"].copy()
    else:
        raise ValueError('Pressure must be reconstructed first')
    p_high = p_high[cells_of_frac_faces]
    
    # USE THE ROTATED COORDINATES TO PERFORM THE EVALUATION OF THE PRESSURE, 
    # BUT USE THE ORIGINAL COORDINATES TO ROTATE THE EDGE USING THE ROTATION
    # MATRIX OF THE LOWER-DIMENSIONAL GRID AS REFERENCE !!!!
        
    # Evaluate the polynomials at the relevant Lagrangian nodes
    point_coo_rot = get_edge_lag_coo(self, gh_rot)
    point_val = utils.eval_P1(p_high, point_coo_rot)
        
    # Rotate the coordinates of the Lagrangian nodes wrt the lower-dim grid
    point_coo = get_edge_lag_coo(self, g_h)
    point_edge_coo_rot = np.empty_like(point_coo)
    for element in range(frac_faces.size):
        point_edge_coo_rot[:, element] = np.dot(R, point_coo[:, element])
    point_edge_coo_rot = point_edge_coo_rot[dim_bool]
    
    # Construct a polynomial (of reduced dimensionality) using the rotated coo
    trace_pressure = utils.interpolate_P1(point_val, point_edge_coo_rot)
    
    # Test if the values of the original polynomial match the new one
    point_val_rot = utils.eval_P1(trace_pressure, point_edge_coo_rot)
    np.testing.assert_almost_equal(point_val, point_val_rot, decimal=12)
    
    return trace_pressure
    

def _get_low_pressure(self, g_l, d_l, frac_cells):
    """
    Obtains the coefficients of the (projected) lower-dimensional pressures

    Parameters
    ----------
    g_l : PorePy object
        Lower-dimensional grid.
    d_l : Dictionary
        Lower-dimensional data dictionary.
    frac_cells : NumPy nd-Array
        Lower-dimensional fracture cells
        
    Raises
    ------
    ValueError
        If the pressure has not been reconstructed
        
    Returns
    -------
    p_low : NumPy nd-Array
        Coefficients of the (projected) lower-dimensional pressures

    """

    # Retrieve lower-dimensional reconstructed pressure coefficients
    if "recon_p" in d_l[self.estimates_kw]:
        p_low = d_l[self.estimates_kw]["recon_p"].copy()
    else:
        raise ValueError("Pressure must be reconstructed first")
    p_low = p_low[frac_cells]
    
    return p_low


def _get_normal_velocity(self, d_e):
    """
    Obtain the normal velocities. That is, the mortar fluxes scaled by the 
    mortar cell measure (area in 2D, length in 1D, one in 0D)

    Parameters
    ----------
    d_e : Dictionary
        Edge data dictionary.
        
    Raises
    ------
    ValueError
        If the mortar fluxes are not in the data dictionary

    Returns
    -------
    normal_velocity : NumPy nd-Array of size (num_mortar_cells, 1)

    """
    
    # Retrieve mortar fluxes from edge dictionary
    if self.lam_name in d_e[pp.STATE]:
        mortar_flux = d_e[pp.STATE][self.lam_name].copy()
    else:
        raise ValueError('Mortar fluxes not found in the data dicitionary')
    
    # Get hold of mortar grid and obtain the volumes of the mortar cells
    mg = d_e["mortar_grid"]
    cell_volumes = mg.cell_volumes
    
    # Obtain the normal velocities and reshape into a column array
    normal_velocity = mortar_flux / cell_volumes
    normal_velocity = normal_velocity.reshape(mortar_flux.size, 1)
    
    return normal_velocity

#%% Interface error [matching]

def _interface_diffusive_error_0d(self, e, d_e):
    """
    Computes interface diffusive flux error for 0D mortar grids

    Parameters
    ----------
    e : PorePy object
        Edge
    d_e : Dictionary
        Edge data dictionary.
        
    Raises
    ------
    ValueError
        If the dimension of the mortar grid is different from zero.
        If the reconstructed pressure is not found in the data dictionaries
        If the mortar flux is not found in the edge dictionary

    Returns
    -------
    diffusive_error : NumPy nd-array
        Diffusive error (squared) for each element of the mortar grid

    """
    
    # Get hold of mortar grids, neighboring grids, and data dicitionaries
    mg = d_e["mortar_grid"]
    g_l, g_h = self.gb.nodes_of_edge(e)
    d_h = self.gb.node_props(g_h)
    d_l = self.gb.node_props(g_l) 
    
    # Run sanity checks
    if mg.dim != 0:
        raise ValueError('Expected zero-dimensional mortar grid')
    if "recon_p" not in d_h[self.estimates_kw]:
        raise ValueError('Pressure must be reconstructed first')
    if "recon_p" not in d_l[self.estimates_kw]:
        raise ValueError('Pressure must be reconstructed first')
    if self.lam_name not in d_e[pp.STATE]:
        raise ValueError('Mortar fluxes not found in the data dictionary')
    
    # Retrieve normal diffusivity. Is this really needed for 0d interfaces?
    normal_diff = d_e[pp.PARAMETERS][self.kw]["normal_diffusivity"]
    if isinstance(normal_diff, int) or isinstance(normal_diff, float):
        k = normal_diff * np.ones([mg.num_cells, 1])
    else:
        k = normal_diff.reshape(mg.num_cells, 1)
        
    # Face-cell map between higher- and lower-dimensional subdomains
    frac_cells, frac_faces, _ = sps.find(d_e["face_cells"])
    
    # Rotate 1d-grid
    gh_rot = utils.rotate_embedded_grid(g_h)  
    
    # Obtain the trace of the pressure of the 1D grid
    cells_of_frac_faces, _, _ = sps.find(g_h.cell_faces[frac_faces].T)      
    p_1D = d_h[self.estimates_kw]["recon_p"].copy()
    p_1D = p_1D[cells_of_frac_faces]
    coo_frac_faces = gh_rot.face_centers[:, frac_faces].T
    coo_frac_faces = coo_frac_faces[np.newaxis, :, :]
    trace_p = utils.eval_P1(p_1D, coo_frac_faces)
    
    # Obtain the pressure of the 0D grid
    p_0D = d_l[self.estimates_kw]["recon_p"].copy()
    p_0D = p_0D[frac_cells]
    
    # Pressure jump
    p_jump = p_0D - trace_p
        
    # Retrieve mortar solution
    mortar_flux = d_e[pp.STATE][self.lam_name].copy()
    normal_vel = mortar_flux / mg.cell_volumes
    normal_vel = normal_vel.reshape(mg.num_cells, 1)
    
    # NOTE: We don't really need to use sidegrids in this case, since 
    # the pressure in 0D domains is unique
    diffusive_error = (k ** (-0.5) * normal_vel + k ** (0.5) * p_jump) ** 2
    
    return diffusive_error  

def _interface_diffusive_error_1d(self, e, d_e):
    """
    Computes diffusive flux error (squared) for one-dimensional mortar grids

    Parameters
    ----------
    e : PorePy Object
        Edge from the Grid Bucket.
    d_e : Dictionary
        Dictionary associated with the edge e.
        
    Raises
    ------
    ValueError
        If the dimension of the mortar grid is different from one.

    Returns
    -------
    diffusive_error: NumPy array of size mg.num_cells
        Diffusive error (squared) for each cell of the mortar grid.

    """

    def compute_sidegrid_error(self, side_tuple):
        """
        This functions projects a mortar quantity to the side grids, and then
        performs the integration on the given side grid.

        Parameters
        ----------
        side_tuple : Tuple
            Containing the sidegrids

        Returns
        -------
        diffusive_error_side : NumPy nd-Array of size (sidegrid.num_cells, 1)
            Diffusive error (squared) for each element of the side grid.

        """
        
        # Get projector and sidegrid object
        projector = side_tuple[0]
        sidegrid = side_tuple[1]
        
        # Rotate side-grid
        sidegrid_rot = utils.rotate_embedded_grid(sidegrid)
        
        # Obtain QuadPy elements
        elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)
        
        # Project relevant quanitites to the side grid
        deltap_side = projector * deltap
        normalvel_side = projector * normal_vel
        k_side = projector * k
        
        # Declare integrand
        def integrand(x):
            coors = x[np.newaxis, :, :] # this is needed for 1D grids
            p_jump = utils.eval_P1(deltap_side, coors)     
            return (
                k_side ** (-0.5) * normalvel_side
                + k_side ** (0.5) * p_jump
                ) ** 2
        
        # Compute integral
        diffusive_error_side = method.integrate(integrand, elements)
        
        return diffusive_error_side
   
    # Get mortar grid and check dimensionality
    mg = d_e["mortar_grid"]
    if mg.dim != 1:
        raise ValueError('Expected one-dimensional mortar grid')
    
    # Get hold of higher- and lower-dimensional neighbors and their dictionaries
    g_l, g_h = self.gb.nodes_of_edge(e)
    d_h = self.gb.node_props(g_h)
    d_l = self.gb.node_props(g_l) 

    # Retrieve normal diffusivity
    normal_diff = d_e[pp.PARAMETERS][self.kw]["normal_diffusivity"]
    if isinstance(normal_diff, int) or isinstance(normal_diff, float):
        k = normal_diff * np.ones([mg.num_cells, 1])
    else:
        k = normal_diff.reshape(mg.num_cells, 1)

    # Face-cell map between higher- and lower-dimensional subdomains
    frac_faces = sps.find(mg.primary_to_mortar_avg().T)[0]
    frac_cells = sps.find(mg.secondary_to_mortar_avg().T)[0]
     
    # Obtain the trace of the higher-dimensional pressure
    tracep_high = _get_high_pressure_trace(self, g_l, g_h, d_h, frac_faces)

    # Obtain the lower-dimensional pressure
    p_low = _get_low_pressure(self, g_l, d_l, frac_cells)
    
    # Now, we can work with the pressure difference
    deltap = p_low - tracep_high
    
    # Obtain normal velocities
    normal_vel = _get_normal_velocity(self, d_e)

    # Declare integration method
    method = qp.c1.newton_cotes_closed(4)
      
    # Retrieve side-grids tuples
    sides = mg.project_to_side_grids()
        
    # Compute the errors for each sidegrid
    diffusive = []
    for side in sides:
        diffusive.append(compute_sidegrid_error(self, side))
    
    # Concatenate into one numpy array
    diffusive_error = np.concatenate(diffusive)
        
    return diffusive_error


def  _interface_diffusive_error_2d(self, e, d_e):
    
    """
    Computes diffusive flux error (squared) for two-dimensional mortar grids

    Parameters
    ----------
    e : PorePy Object
        Edge from the Grid Bucket.
    d_e : Dictionary
        Dictionary associated with the edge e.
        
    Raises
    ------
    ValueError
        If the dimension of the mortar grid is different from two.

    Returns
    -------
    diffusive_error: NumPy array of size mg.num_cells
        Diffusive error (squared) for each cell of the mortar grid.

    """

    def compute_sidegrid_error(self, side_tuple):
        """
        This functions projects a mortar quantity to the side grids, and then
        performs the integration on the given side grid.

        Parameters
        ----------
        side_tuple : Tuple
            Containing the sidegrids

        Returns
        -------
        diffusive_error_side : NumPy nd-Array of size (sidegrid.num_cells, 1)
            Diffusive error (squared) for each element of the side grid.

        """
        
        # Get projector and sidegrid object
        projector = side_tuple[0]
        sidegrid = side_tuple[1]
        
        # Rotate side-grid
        sidegrid_rot = utils.rotate_embedded_grid(sidegrid)
        
        # Obtain QuadPy elements
        elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)
        
        # Project relevant quanitites to the side grid
        deltap_side = projector * deltap
        normalvel_side = projector * normal_vel
        k_side = projector * k
        
        # Declare integrand
        def integrand(x):
            p_jump = utils.eval_P1(deltap_side, x)     
            return (
                k_side ** (-0.5) * normalvel_side
                + k_side ** (0.5) * p_jump
                ) ** 2
        
        # Compute integral
        diffusive_error_side = method.integrate(integrand, elements)
        
        return diffusive_error_side
   
    # Get mortar grid and check dimensionality
    mg = d_e["mortar_grid"]
    if mg.dim != 2:
        raise ValueError('Expected two-dimensional mortar grid')
    
    # Get hold of higher- and lower-dimensional neighbors and their dictionaries
    g_l, g_h = self.gb.nodes_of_edge(e)
    d_h = self.gb.node_props(g_h)
    d_l = self.gb.node_props(g_l) 

    # Retrieve normal diffusivity
    normal_diff = d_e[pp.PARAMETERS][self.kw]["normal_diffusivity"]
    if isinstance(normal_diff, int) or isinstance(normal_diff, float):
        k = normal_diff * np.ones([mg.num_cells, 1])
    else:
        k = normal_diff.reshape(mg.num_cells, 1)

    # Face-cell map between higher- and lower-dimensional subdomains
    frac_faces = sps.find(mg.primary_to_mortar_avg().T)[0]
    frac_cells = sps.find(mg.secondary_to_mortar_avg().T)[0]
     
    # Obtain the trace of the higher-dimensional pressure
    tracep_high = _get_high_pressure_trace(self, g_l, g_h, d_h, frac_faces)

    # Obtain the lower-dimensional pressure
    p_low = _get_low_pressure(self, g_l, d_l, frac_cells)
    
    # Now, we can work with the pressure difference
    deltap = p_low - tracep_high
    
    # Obtain normal velocities
    normal_vel = _get_normal_velocity(self, d_e)

    # Declare integration method
    method = qp.t2.get_good_scheme(3)
      
    # Retrieve side-grids tuples
    sides = mg.project_to_side_grids()
        
    # Compute the errors for each sidegrid
    diffusive = []
    for side in sides:
        diffusive.append(compute_sidegrid_error(self, side))
    
    # Concatenate into one numpy array
    diffusive_error = np.concatenate(diffusive)
        
    return diffusive_error

#%% Interface error [non-matching]
def _mortar_highdim_faces_mapping(mg, side):
    """
    Get mortar cells - high-dim fracture faces mapping for a given interface side

    Parameters
    ----------
    mg : PorePy Object
        Mortar Grid.
    side : Integer
        Side of the interface, either -1 or 1.

    Returns
    -------
    mortar_highfaces_side_map : NumPy nd-array of size (2 x number of maps)
        Array containing the mappings. Note that for the case of non-matching
        grids, the mappings are not unique.
    """
    
    # General mapping
    mortar_highfaces = mg.primary_to_mortar_avg().nonzero()
    # Signs of the mortar cells
    mortar_signs = sps.find(mg.sign_of_mortar_sides())[2]
    # Construct the output array for the given side    
    mortar_highfaces_side_map = np.array(
        [mortar_highfaces[0][mortar_signs[mortar_highfaces[0]] == side],
         mortar_highfaces[1][mortar_signs[mortar_highfaces[0]] == side]
         ])
    
    return mortar_highfaces_side_map
    
    
def _mortar_lowdim_cells_mapping(mg, side):
    """
    Get mortar cells - low-dim fracture cells mapping for a given interface side

    Parameters
    ----------
    mg : PorePy Object
        Mortar Grid.
    side : Integer
        Side of the interface, either -1 or 1.

    Returns
    -------
    mortar_lowcells_side_map : NumPy nd-array of size (2 x number of maps)
        Array containing the mappings. Note that for the case of non-matching
        grids, the mappings are not unique.
    """
    
    # General mapping
    mortar_lowcells = mg.secondary_to_mortar_avg().nonzero()
    # Signs of the mortar cells
    mortar_signs = sps.find(mg.sign_of_mortar_sides())[2]
    # Construct the output array for the given side    
    mortar_lowcells_side_map = np.array(
        [mortar_lowcells[0][mortar_signs[mortar_lowcells[0]] == side],
         mortar_lowcells[1][mortar_signs[mortar_lowcells[0]] == side]
         ])
    
    return mortar_lowcells_side_map


def _sorted_highdim_edge_grid(g_h, g_l, mg, side):
    """
    Creates a sorted, rotated pseudo-grid from the high-dimensional fracture
    faces nodes that are adjacent to the given side of the mortar grid

    Parameters
    ----------
    g_h : PorePy object
        High-dimensional grid.
    g_l : PorePy object
        Low-dimensional grid.
    mg : PorePy object 
        Mortar Grid.
    side : Integer
        Either -1 or 1.

    Raises
    ------
    ValueError
        If the dimensionality of the mortar grid is different from one.

    Returns
    -------
    rot_frac_faces_nodes_coo : NumPy array of size (frac_faces x 2)
        Rotated and sorted fracture faces nodes coordinates.
    sorted_frac_faces : List of length frac_faces
        Sorted fracture faces (global) indices.
        
    """
    
    # Sanity checks
    if mg.dim != 1:
        raise ValueError('Expected one-dimensional mortar grid')
    
    # Get hold of mortar cells and higher-dim faces mapping
    mortar_highfaces_side_map = _mortar_highdim_faces_mapping(mg, side)
    # Since a higher-dimensional face can be connected with more than
    # one mortar cell, we need to extract the unique faces
    frac_faces = np.unique(mortar_highfaces_side_map[1])
    # Now we need to sort the high-dim faces according to their coordinates
    # We use the face-centers as a reference   
    frac_faces_cc = g_h.face_centers[:, frac_faces]
    # Now, we need to rotate to the face centers. For the purpose, we 
    # use the rotation matrix of the lower-dimensional grid
    gl_rot = utils.rotate_embedded_grid(g_l) # rotate low-dim grid
    R = gl_rot.rotation_matrix # rotation matrix
    rot_frac_faces_minus_cc = np.dot(R, frac_faces_cc)
    # We're only interested in the active dimension
    rot_frac_faces_minus_cc = rot_frac_faces_minus_cc[gl_rot.dim_bool]
    # Now, we sort the fracture faces according to their (rotate) face 
    # center coordinates. We actually only need the indices
    sorted_idx = np.argsort(rot_frac_faces_minus_cc).flatten()
    # Sorted higher-dimensional fracture faces
    sorted_frac_faces = frac_faces[sorted_idx]
    
    # Now that we have the sorted fracture faces, we can extract their nodes
    # and construct a FEM 1D-grid
    sorted_frac_faces_nodes  = np.reshape(
        sps.find(g_h.face_nodes.T[sorted_frac_faces].T)[0], 
        [sorted_frac_faces.size, g_h.dim]
        )
    sorted_frac_faces_nodes_coo = g_h.nodes[:, sorted_frac_faces_nodes]
    
    # Perform the rotation, and stack the coordinates
    rot_frac_faces_nodes_coo_0 = np.dot(R, sorted_frac_faces_nodes_coo[:, :, 0])
    rot_frac_faces_nodes_coo_0 = rot_frac_faces_nodes_coo_0[gl_rot.dim_bool]
    rot_frac_faces_nodes_coo_1 = np.dot(R, sorted_frac_faces_nodes_coo[:, :, 1])
    rot_frac_faces_nodes_coo_1 = rot_frac_faces_nodes_coo_1[gl_rot.dim_bool]
    
    rot_frac_faces_nodes_coo = np.array([
        rot_frac_faces_nodes_coo_0.flatten(),
        rot_frac_faces_nodes_coo_1.flatten()
        ]).T
    
    # We apply one more sorting, to have also locally sorted elements
    rot_frac_faces_nodes_coo = np.sort(rot_frac_faces_nodes_coo)
    
    return rot_frac_faces_nodes_coo, sorted_frac_faces
    

def _sorted_side_grid(mg, g_l, side):
    """
    Creates a sorted, rotated pseudo-grid from the nodes composing the mortar
    cells of the given side of the mortar grid

    Parameters
    ----------
    mg : PorePy object 
        Mortar Grid.
    g_l : PorePy object
        Low-dimensional grid.
    side : Integer
        Either -1 or 1.

    Raises
    ------
    ValueError
        If the dimensionality of the mortar grid is different from one.

    Returns
    -------
    rot_mortar_cells_nodes_coo : NumPy array of size (mortar_cells x 2)
        Rotated and sorted mortar cells nodes coordinates.
    sorted_mortar_cells : List of length mortar cells
        Sorted mortar cells (global) indices.

    """
    
    # Sanity checks
    if mg.dim != 1:
        raise ValueError('Expected one-dimensional mortar grid')
    
    # Get the mortar cells corresponding to the given side
    signs_idx = sps.find(mg.sign_of_mortar_sides())[2] == side
    mortar_cells = np.arange(mg.num_cells)[signs_idx]
    
    # Retrieve the correct side grid according to the given side
    for side_obj in mg.project_to_side_grids():
        projected_mortar_cells = side_obj[0].nonzero()[1]
        if (mortar_cells - projected_mortar_cells).sum() == 0:
            side_grid = side_obj[1]
    
    # We will sort the mortar cells using the cell centers as reference
    # NOTE: I'm not sure if this is necessary, but it does not harm to do it
    sg_cc = side_grid.cell_centers # retrieve cell-centers    
    gl_rot = utils.rotate_embedded_grid(g_l) # rotate low-dim grid
    R = gl_rot.rotation_matrix # extract rotation matrix
    rot_sg_cc = np.dot(R, sg_cc) # rotate cell-centers
    rot_sg_cc = rot_sg_cc[gl_rot.dim_bool] # we need only the active dim
    # Obtain the indices of the sorted mortar cells
    sorted_idx = np.argsort(rot_sg_cc).flatten()
    # Sort the mortar cells
    sorted_mortar_cells = mortar_cells[sorted_idx]
    
    # Now that we have the sorted mortar cells, we can extract their nodes
    # and construct a FEM 1D-grid
    sorted_mortar_cells_nodes  = np.reshape(
        sps.find(side_grid.cell_nodes().T[np.arange(mortar_cells.size)].T)[0], 
        [sorted_mortar_cells.size, side_grid.dim + 1]
        )
    sorted_mortar_cells_nodes_coo = side_grid.nodes[:, sorted_mortar_cells_nodes]
    
    # Perform the rotation, and stack the coordinates
    rot_mortar_cells_nodes_coo_0 = np.dot(R, sorted_mortar_cells_nodes_coo[:, :, 0])
    rot_mortar_cells_nodes_coo_0 = rot_mortar_cells_nodes_coo_0[gl_rot.dim_bool]
    rot_mortar_cells_nodes_coo_1 = np.dot(R, sorted_mortar_cells_nodes_coo[:, :, 1])
    rot_mortar_cells_nodes_coo_1 = rot_mortar_cells_nodes_coo_1[gl_rot.dim_bool]
    rot_mortar_cells_nodes_coo = np.array([
        rot_mortar_cells_nodes_coo_0.flatten(),
        rot_mortar_cells_nodes_coo_1.flatten()
        ]).T
    
    # We apply one more sorting, to have also locally sorted elements   
    rot_mortar_cells_nodes_coo = np.sort(rot_mortar_cells_nodes_coo)
        
    return rot_mortar_cells_nodes_coo, sorted_mortar_cells


def _sorted_low_grid(g_l):
    """
    Creates a sorted, rotated pseudo-grid from the nodes composing the lower-
    dimensional cells. Note that no notion of sides should be prescribed, since
    the lower-dimensional grid is uniquely coupled to both sides of an interface

    Parameters
    ----------
    g_l : PorePy object
        Lower-dimensional grid.

    Raises
    ------
    ValueError
        If the dimensionality of the grid is different from one

    Returns
    -------
    rot_low_cells_nodes_coo : NumPy array of size (low_cells x 2)
        Rotated and sorted lower-diemensional cells nodes coordinates.
    sorted_low_cells : List of length lower-dimensional cells
        Sorted lower-dimensional cells (global) indices.

    """

    # Sanity check
    if g_l.dim != 1:
        raise ValueError('Expected one-dimensional grid')

    # The cells of the lower-dimensional grid do not have sides
    low_cells = np.arange(g_l.num_cells)
    # We will sort the cells using the cell centers as a reference
    gl_rot = utils.rotate_embedded_grid(g_l)
    R = gl_rot.rotation_matrix
    sorted_idx = np.argsort(gl_rot.cell_centers).flatten()
    sorted_low_cells = low_cells[sorted_idx]
    
    # Now that we have the sorted low-dim cells, we can extract their nodes
    # and construct a FEM 1D-grid
    sorted_low_cells_nodes = np.reshape(
        sps.find(g_l.cell_nodes().T[sorted_low_cells].T)[0],
        [sorted_low_cells.size, g_l.dim + 1]
        )
    sorted_low_cells_nodes_coo = g_l.nodes[:, sorted_low_cells_nodes]
    
    # Perform the rotation, and stack the coordinates
    rot_low_cells_nodes_coo_0 = np.dot(R, sorted_low_cells_nodes_coo[:, :, 0])
    rot_low_cells_nodes_coo_0 = rot_low_cells_nodes_coo_0[gl_rot.dim_bool]
    rot_low_cells_nodes_coo_1 = np.dot(R, sorted_low_cells_nodes_coo[:, :, 1])
    rot_low_cells_nodes_coo_1 = rot_low_cells_nodes_coo_1[gl_rot.dim_bool]
    rot_low_cells_nodes_coo = np.array([
        rot_low_cells_nodes_coo_0.flatten(),
        rot_low_cells_nodes_coo_1.flatten()
    ]).T

    return rot_low_cells_nodes_coo, sorted_low_cells

def _merge_grids(low_grid, mortar_grid, high_grid):
    """
    Unifies lower-dimensional, mortar, and higher-dimensional grids into one

    Parameters
    ----------
    low_grid : NumPy Array of size [num_low_cells x 2]
        Lower-dimensional pseudo-grid.
    mortar_grid : NumPy Array of size [num_side_mortar_cells x 2]
        Side pseudo-grid.
    high_grid : NumPy Array of size [num_side_frac_faces x 2]
        Higher-dimensional internal sided-boundary pseudo-grid

    Returns
    -------
    merged_grid : NumPy Array of size [num_merged_elements x 2]
        Merged grid containing the nodes of the merged elements

    """
    
    # Merge higher-dimensional and mortar (side) grid
    high_union_mortar = np.array([
        np.union1d(high_grid[:, 0], mortar_grid[:, 0]),
        np.union1d(high_grid[:, 1], mortar_grid[:, 1]),
        ]).T
    
    # Merge the above grid with the lower-dimensional grid
    merged_grid = np.array([
         np.union1d(high_union_mortar[:, 0], low_grid[:, 0]),
         np.union1d(high_union_mortar[:, 1], low_grid[:, 1]),
        ]).T
    
    return merged_grid


def _get_grid_uniongrid_elements(merged_grid, grid):
    """
    Get the mapping between a grid and the merged grid

    Parameters
    ----------
    merged_grid : NumPy Array of size [num_merged_elements x 2]
        Merged pseudo-grid, i.e., as obtained with merge_grids()
    grid : NumPy Array of size [num_grid_elements x 2]
        Pseugo grid, i.e.: high_grid, mortar_grid, or low_grid.

    Returns
    -------
    elements : List of length num_merged_elements
        Containing the local elements of the grids overlapping the merged grid,
        at the a interval.

    Credits
    -------
    The follwing piece of code was modified from:
    www.geeksforgeeks.org/find-intersection-of-intervals-given-by-two-lists/
    Author: Sarthak Shukla (Indian Institute of Information Technology Nagpur)
    
    """
    
    
    # First, convert the FEM-like 1D grids to lists
    arr1 = merged_grid.tolist()
    arr2 = grid.tolist()
    
    # Initialize pointers
    i = j = 0

    # Length of lists
    n = len(arr1) 
    m = len(arr2) 
    
    # Elements list
    elements = []
    
    # Loop through all intervals unless one of the interval gets exhausted 
    while i < n and j < m: 
      
        # Left bound for intersecting segment 
        l = max(arr1[i][0], arr2[j][0]) 
          
        # Right bound for intersecting segment 
        r = min(arr1[i][1], arr2[j][1]) 
        
        # If the segment is valid, append the element to the list
        if l < r:
            elements.append(j)
            
        # If i-th interval's right bound is  
        # smaller increment i else increment j 
        if arr1[i][1] < arr2[j][1]: 
            i += 1
        else: 
            j += 1
 
    return elements


def _project_poly_to_merged_grid(self, e, d_e, sorted_elements, merged_grid_map):
    """
    Projects grid quantities (pressures, normal permeabilities, and
    normal_velocities) to the merged grid

    Parameters
    ----------
    e : PorePy object
        Edge.
    d_e : Dictionary
        Edge dictionary.
    sorted_elements : List of length 3
        Containing the sorted (global) indices of the grid elements. The list
        must be passed using the following ordering:
            [sorted_low_cells, sorted_mortar_cells, sorted_high_faces]
    merged_grid_map : List of length 3
        Containing the (local) mapping between the grid and the merged grid. he list
        must be passed using the following ordering:
            [low_mapping, mortar_mapping, high_mapping]

    Returns
    -------
    p_jump_merged : NumPy Array of size [num_merged_elements x 2]
        Containing the projected P1 local polynomials describing the pressure 
        jump. That is p_low - trace_p_high
    k_perp : NumPy Array of size [num_merged_elements x 1]
        Projection of normal diffusivities onto the merged grid.
    normal_vel_merged : NumPy Array of size [num_merged_elements x 1]
        Projection of normal velocities onto the merged grid
    """
    
    # Retrieve neighboring cells and data dictionaries
    g_l, g_h = self.gb.nodes_of_edge(e)
    d_h = self.gb.node_props(g_h)
    d_l = self.gb.node_props(g_l) 
    
    # Retrieve the sorted elements for each grid
    low_cells = sorted_elements[0]
    mortar_cells = sorted_elements[1]
    frac_faces = sorted_elements[2]
    
    # Get the elements mapping the merged grid
    merged_low_elements = merged_grid_map[0]
    merged_mortar_elements = merged_grid_map[1]
    merged_high_elements = merged_grid_map[2]
    
    # Get hold of pressure trace polynomials
    tracep_high = _get_high_pressure_trace(self, g_l, g_h, d_h, frac_faces)
    tracep_high_merged = tracep_high[merged_high_elements]
    
    # Get hold of lower-dimensional pressure polynomials
    p_low = _get_low_pressure(self, g_l, d_l, low_cells)
    p_low_merged = p_low[merged_low_elements]
    
    # We can now work with the pressure jump
    p_jump_merged = p_low_merged - tracep_high_merged
    
    # Get hold of normal permeabilities
    normal_diff = d_e[pp.PARAMETERS][self.kw]["normal_diffusivity"]
    if isinstance(normal_diff, int) or isinstance(normal_diff, float):
        k_perp = normal_diff * np.ones([len(merged_mortar_elements), 1])
    else:
        k_perp = normal_diff[mortar_cells[merged_mortar_elements]]
        k_perp = k_perp.reshape(len(merged_mortar_elements), 1)
    
    # Get hold of normal velocities
    normal_vel = _get_normal_velocity(self, d_e)
    normal_vel_merged = normal_vel[mortar_cells[merged_mortar_elements]]
    
    return p_jump_merged, k_perp, normal_vel_merged


def _interface_diffusive_error_nonmatching_1d(self, e, d_e):
    """
    Computes the diffusive error (squared) for the entire mortar grid. This 
    function should be used when there exists a non-matching coupling between
    the grids and the interfaces. If the coupling involves matching grids,
    interface_diffusive_error_1d() should be used. However, in principle, the 
    output should be the same.

    Parameters
    ----------
    e : PorePy object
        Edge.
    d_e : Dictionary
        Interface dictionary.

    Returns
    -------
    diffusive_error: NumPy Array of size mg.num_cells
        Diffusive error (squared) incurred on each mortar cell.

    """
    
    # Get hold of grids and dictionaries
    g_l, g_h = self.gb.nodes_of_edge(e)
    mg = d_e['mortar_grid']
    
    # Obtain the number of sides of the mortar grid
    num_sides = mg.num_sides()
    if num_sides == 2:
        sides = [-1, 1]
    else:
        sides = [1]
        
    # Loop over the sides of the mortar grid
    diffusive_error = np.zeros(mg.num_cells)
    
    for side in sides:
        
        # Get rotated grids and sorted elements
        high_grid, frac_faces = _sorted_highdim_edge_grid(g_h, g_l, mg, side)
        mortar_grid, mortar_cells = _sorted_side_grid(mg, g_l, side)
        low_grid, low_cells = _sorted_low_grid(g_l)
        
        # Merge the three grids into one
        merged_grid = _merge_grids(low_grid, mortar_grid, high_grid)
    
        # Note that the following mappings are local for each merged grid.
        # For example, to retrieve the global fracture faces indices, we should
        # write frac_faces[merged_high_ele], and to retrieve the global mortar
        # cells, we should write mortar_cells[merged_mortar_ele]
        # Retrieve element mapping from sorted grids to merged grid
        merged_high_ele = _get_grid_uniongrid_elements(merged_grid, high_grid)
        merged_mortar_ele =  _get_grid_uniongrid_elements(merged_grid, mortar_grid)
        merged_low_ele = _get_grid_uniongrid_elements(merged_grid, low_grid)

        # Get projected pressure jump, normal permeabilities, and normal velocities
        pressure_jump, k_perp, normal_vel = _project_poly_to_merged_grid(
            self, e, d_e, 
            [low_cells, mortar_cells, frac_faces], 
            [merged_low_ele, merged_mortar_ele, merged_high_ele]
            )    
    
        # Define integration method and obtain quadpy elements
        method = qp.c1.newton_cotes_closed(4)
        qp_ele = utils.get_qp_elements_from_union_grid_1d(merged_grid)

        # Define integrand
        def integrand(x):
            coors = x[np.newaxis, :, :] # this is needed for 1D grids
            p_jump = utils.eval_P1(pressure_jump, coors) # eval pressure jump
            return (
                k_perp ** (-0.5) * normal_vel
                + k_perp ** (0.5) * p_jump
                ) ** 2
        
        # Evaluate integral
        diffusive_error_merged = method.integrate(integrand, qp_ele)
        
        # Sum errors corresponding to a mortar cell
        diffusive_error_side = np.zeros(len(mortar_cells))
        for mortar_element in range(len(mortar_cells)):
            idx = mortar_cells[mortar_element] == mortar_cells[merged_mortar_ele]
            diffusive_error_side[mortar_element] = diffusive_error_merged[idx].sum()
    
        # Append into the list
        diffusive_error[mortar_cells] = diffusive_error_side
        
    return diffusive_error

    
#%% L2- error estimates (for testing purposes)

def l2_velocity(g, d, idx_top, idx_middle, idx_bot):

    # Rotate grid
    g_rot = utils.rotate_embedded_grid(g)

    # Retrieve elements
    elements = utils.get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.c1.newton_cotes_closed(4)
    elif g.dim == 2:
        method = qp.t2.strang_fix_cowper_05()

    # Retrieve velocity coefficients
    uh = d["estimates"]["recon_u"].copy()

    # Quadpyfying arrays
    if g.dim == 1:
        a = uh[:, 0].reshape(uh.shape[0], 1)
        b = uh[:, 1].reshape(uh.shape[0], 1)
    elif g.dim == 2:
        a = uh[:, 0].reshape(uh.shape[0], 1)
        b = uh[:, 1].reshape(uh.shape[0], 1)
        c = uh[:, 2].reshape(uh.shape[0], 1)

    # Define integration regions for 2D subdomain
    def top_subregion(X):
        x = X[0]
        y = X[1]

        uexa_x = -(x - 0.5) / ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** (0.5)
        uexa_y = -(y - 0.75) / ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** (0.5)
        urec_x = a * x + b
        urec_y = a * y + c

        int_x = (uexa_x - urec_x) ** 2
        int_y = (uexa_y - urec_y) ** 2

        return int_x + int_y

    def mid_subregion(X):
        x = X[0]
        y = X[1]

        uexa_x = -(((x - 0.5) ** 2) ** (0.5)) / (x - 0.5)
        uexa_y = 0
        urec_x = a * x + b
        urec_y = a * y + c

        int_x = (uexa_x - urec_x) ** 2
        int_y = (uexa_y - urec_y) ** 2

        return int_x + int_y

    def bot_subregion(X):
        x = X[0]
        y = X[1]

        uexa_x = -(x - 0.5) / ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** (0.5)
        uexa_y = -(y - 0.25) / ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** (0.5)
        urec_x = a * x + b
        urec_y = a * y + c

        int_x = (uexa_x - urec_x) ** 2
        int_y = (uexa_y - urec_y) ** 2

        return int_x + int_y

    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        uexa_x = 0
        urec_x = a * x + b

        int_x = (uexa_x - urec_x) ** 2

        return int_x

    # Compute errors
    if g.dim == 2:
        int_top = method.integrate(top_subregion, elements)
        int_mid = method.integrate(mid_subregion, elements)
        int_bot = method.integrate(bot_subregion, elements)
        integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
    elif g.dim == 1:
        integral = method.integrate(fracture, elements)

    return np.sqrt(integral.sum())


def l2_reconp(g, d, idx_top, idx_middle, idx_bot):

    # Rotate grid
    g_rot = utils.rotate_embedded_grid(g)

    # Retrieve elements
    elements = utils.get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.c1.newton_cotes_closed(4)
    elif g.dim == 2:
        method = qp.t2.strang_fix_cowper_05()

    # Retrieve postprocess pressure
    ph = d["estimates"]["recon_p"].copy()
    c = utils.poly2col(ph)

    # Define fracture regions for the bulk
    def top_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5
        p_rec = c[0] * x + c[1] * y + c[2]
        int_ = (p_ex - p_rec) ** 2

        return int_

    def mid_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2) ** 0.5
        p_rec = c[0] * x + c[1] * y + c[2]
        int_ = (p_ex - p_rec) ** 2

        return int_

    def bot_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
        p_rec = c[0] * x + c[1] * y + c[2]
        int_ = (p_ex - p_rec) ** 2

        return int_

    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        p_rec = c[0] * x + c[1]

        int_ = (p_ex - p_rec) ** 2

        return int_

    # Compute errors
    if g.dim == 2:
        int_top = method.integrate(top_subregion, elements)
        int_mid = method.integrate(mid_subregion, elements)
        int_bot = method.integrate(bot_subregion, elements)
        integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
    elif g.dim == 1:
        integral = method.integrate(fracture, elements)

    return np.sqrt(integral.sum())


def l2_gradreconp(g, d, idx_top, idx_middle, idx_bot):
    
    # Rotate grid
    g_rot = utils.rotate_embedded_grid(g)

    # Retrieve elements
    elements = utils.get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.c1.newton_cotes_closed(4)
    elif g.dim == 2:
        method = qp.t2.strang_fix_cowper_05()

    # Retrieve postprocess pressure
    ph = d["estimates"]["recon_p"].copy()
    c = utils.poly2col(ph)
    
    # Define fracture regions for the bulk
    def top_subregion(x):
        grad_pex_x = (x[0] - 0.5) / ((x[0] - 0.5) ** 2 + (x[1] - 0.75) ** 2) ** (0.5)
        grad_pex_y = (x[1] - 0.75) / ((x[0] - 0.5) ** 2 + (x[1]- 0.75) ** 2) ** (0.5)              
               
        grad_prec_x = c[0] * np.ones_like(x[0])
        grad_prec_y = c[1] * np.ones_like(x[1])
        
        int_x = (grad_pex_x - grad_prec_x) ** 2
        int_y = (grad_pex_y - grad_prec_y) ** 2

        return int_x + int_y

    def mid_subregion(x):
        grad_pex_x = ((x[0] - 0.5) ** 2) ** (0.5) / (x[0] - 0.5)
        grad_pex_y = 0    

        grad_prec_x = c[0] * np.ones_like(x[0])
        grad_prec_y = c[1] * np.ones_like(x[1])
        
        int_x = (grad_pex_x - grad_prec_x) ** 2
        int_y = (grad_pex_y - grad_prec_y) ** 2

        return int_x + int_y

    def bot_subregion(x):
        grad_pex_x = (x[0] - 0.5) / ((x[0] - 0.5) ** 2 + (x[1] - 0.25) ** 2 ) ** (0.5)
        grad_pex_y = (x[1] - 0.25) / ((x[0] - 0.5) ** 2 + (x[1]- 0.25) ** 2) ** (0.5)                


        grad_prec_x = c[0] * np.ones_like(x[0])
        grad_prec_y = c[1] * np.ones_like(x[1])
        
        int_x = (grad_pex_x - grad_prec_x) ** 2
        int_y = (grad_pex_y - grad_prec_y) ** 2
        
        return int_x + int_y


    # Define integration regions for the fracture
    def fracture(x):

        grad_pex_x = 0
        
        grad_prec_x = c[0] * np.ones_like(x[0])

        int_x = (grad_pex_x - grad_prec_x) ** 2

        return int_x

    # Compute errors
    if g.dim == 2:
        int_top = method.integrate(top_subregion, elements)
        int_mid = method.integrate(mid_subregion, elements)
        int_bot = method.integrate(bot_subregion, elements)
        integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
    elif g.dim == 1:
        integral = method.integrate(fracture, elements)

    return np.sqrt(integral.sum())


def l2_tracep(gb, e, d_e):
    
    
    def _get_tracep_high_1d(g_l, g_h, d_h, frac_faces):
        """
        Obtains the coefficients of the (projected) trace of the pressure
        
        Parameters
        ----------
        g_l : PorePy object
            Lower-dimensional grid.
        g_h : PorePy object
            Higher-dimensional grid.
        d_h : Dictionary
            Higher-dimensional data dictionary.
        frac_faces : NumPy nd-array
            Higher-dimensional fracture faces
        which_p : string
            Either "recon_p" or "postp_p"
    
        Returns
        -------
        trace_pressure : NumPy nd-array
            Coefficients of the trace of the pressure
    
        """
        def get_edge_lag_coo(grid):
            nodes_of_frac_faces = np.reshape(
                sps.find(g_h.face_nodes.T[frac_faces].T)[0], 
                [frac_faces.size, g_h.dim]
                )
            nodes_coo_of_frac_faces = grid.nodes[:, nodes_of_frac_faces]
            lagrangian_coo = nodes_coo_of_frac_faces
            return lagrangian_coo
       
        # Rotate both grids, and obtain rotation matrix and effective dimension
        gh_rot = utils.rotate_embedded_grid(g_h)
        gl_rot = utils.rotate_embedded_grid(g_l)
        R = gl_rot.rotation_matrix
        dim_bool = gl_rot.dim_bool
        
        # Obtain the cells coorresponding to the frac_faces
        cells_of_frac_faces, _, _ = sps.find(g_h.cell_faces[frac_faces].T)      
        
        # Retrieve the coefficients of the polynomials corresponding to those cells
        p_high = d_h["estimates"]["recon_p"].copy()
        p_high = p_high[cells_of_frac_faces]
        
        # USE THE ROTATED COORDINATES TO PERFORM THE EVALUATION OF THE PRESSURE, 
        # BUT USE THE ORIGINAL COORDINATES TO ROTATE THE EDGE USING THE ROTATION
        # MATRIX OF THE LOWER-DIMENSIONAL GRID AS REFERENCE !!!!
            
        # Evaluate the polynomials at the relevant Lagrangian nodes
        point_coo_rot = get_edge_lag_coo(gh_rot)
        point_val = utils.eval_P1(p_high, point_coo_rot)
            
        # Rotate the coordinates of the Lagrangian nodes wrt the lower-dim grid
        point_coo = get_edge_lag_coo(g_h)
        point_edge_coo_rot = np.empty_like(point_coo)
        for element in range(frac_faces.size):
            point_edge_coo_rot[:, element] = np.dot(R, point_coo[:, element])
        point_edge_coo_rot = point_edge_coo_rot[dim_bool]
        
        # Construct a polynomial (of reduced dimensionality) using the rotated coo
        trace_pressure = utils.interpolate_P1(point_val, point_edge_coo_rot)
        
        # Test if the values of the original polynomial match the new one
        point_val_rot = utils.eval_P1(trace_pressure, point_edge_coo_rot)
        np.testing.assert_almost_equal(point_val, point_val_rot, decimal=12)
        
        return trace_pressure
    
    
    def compute_sidegrid_trace_error(side_tuple):
        """
        This functions projects a mortar quantity to the side grids, and then
        performs the integration

        Parameters
        ----------
        side_tuple : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Get projector and sidegrid object
        projector = side_tuple[0]
        sidegrid = side_tuple[1]
        # Rotate side-grid
        sidegrid_rot = utils.rotate_embedded_grid(sidegrid)
        # Obtain QuadPy elements
        elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)
        # Project normal velocities, pressure jump, and normal diffusivities
        tracep_side = projector * tracep_high
        # Declare integrand
        def integrand(x):
            coors = x[np.newaxis, :, :]
            tracep = utils.eval_P1(tracep_side, coors)     
            return (0 - tracep) ** 2
        # Compute integral
        error = method.integrate(integrand, elements)
        return error
    
    # Get grids and data dictionaries
    mg = d_e["mortar_grid"]
    g_l, g_h = gb.nodes_of_edge(e)
    d_h = gb.node_props(g_h)

    # Face-cell map between higher- and lower-dimensional subdomains
    frac_faces = sps.find(mg.master_to_mortar_avg().T)[0]
     
    # Obtain the trace of the higher-dimensional pressure
    tracep_high = _get_tracep_high_1d(g_l, g_h, d_h, frac_faces)
  
    # Declare integration method
    method = qp.c1.newton_cotes_closed(4)
      
    # Retrieve side-grids tuples
    sides = mg.project_to_side_grids()
    
    # Compute the errors for each sidegrid
    side_error = []
    for side in sides:
        side_error.append(compute_sidegrid_trace_error(side))
    
    # Concatenate into one numpy array
    trace_error = np.concatenate(side_error)
        
    return np.sqrt(trace_error.sum())
    

def l2_mortar(gb, e, d_e):
    

    def _get_normal_vel(d_e, g_h, frac_faces):
        """
        Obtain the normal velocities, that is the mortar fluxes / frac areeas
        
        Parameters
        ----------
        d_e : Dictionary
            Edge data dictionary.
        g_h : PorePy object
            Higher-dimesnional grid.
        frac_faces : NumPy nd-Array
            Higher-dimensional fracture faces.
        
        Returns
        -------
        normal_vel : NumPy nd-Array
            Normal velocities.
        
        """
        
        mortar_flux = d_e[pp.STATE]["interface_flux"].copy()
        frac_faces_areas = g_h.face_areas[frac_faces]
        normal_vel = mortar_flux / frac_faces_areas
        normal_vel = normal_vel.reshape(mortar_flux.size, 1)
        
        return normal_vel

    def compute_sidegrid_mortar_error(side_tuple):
        """
        This functions projects a mortar quantity to the side grids, and then
        performs the integration

        Parameters
        ----------
        side_tuple : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Get projector and sidegrid object
        projector = side_tuple[0]
        sidegrid = side_tuple[1]
        # Rotate side-grid
        sidegrid_rot = utils.rotate_embedded_grid(sidegrid)
        # Obtain QuadPy elements
        elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)
        # Project normal velocities, pressure jump, and normal diffusivities
        mortar_side = projector * normal_vel
        # Declare integrand
        def integrand(x):
            coors = x[np.newaxis, :, :]
            return (1 - mortar_side * np.ones_like(coors)) ** 2
        # Compute integral
        error = method.integrate(integrand, elements)
        return error


    # Get grids and data dictionaries
    mg = d_e["mortar_grid"]
    g_l, g_h = gb.nodes_of_edge(e)

    # Face-cell map between higher- and lower-dimensional subdomains
    frac_faces = sps.find(mg.master_to_mortar_avg().T)[0]
       
    # Obtain normal velocities
    normal_vel = _get_normal_vel(d_e, g_h, frac_faces)

    # Declare integration method
    method = qp.c1.newton_cotes_closed(4)
      
    # Retrieve side-grids tuples
    sides = mg.project_to_side_grids()
        
    # Compute the errors for each sidegrid
    error_mortar = []
    for side in sides:
        error_mortar.append(compute_sidegrid_mortar_error(side))
    
    # Concatenate into one numpy array
    error = np.concatenate(error_mortar)
        
    return np.sqrt(error.sum())


def l2_cc_postp_p(g, d, kw, true_pcc):

    # First, rotate grid
    g_rot = utils.rotate_embedded_grid(g)

    # Obtain volumes and cell centers
    V = g.cell_volumes
    cc = g_rot.cell_centers

    # Retrieve post process pressure
    postp_coeff = d["error_estimates"]["recon_p"].copy()

    # Evaluate at the cell_centers
    postp_cc = postp_coeff[:, 0]
    for dim in range(g.dim):
        postp_cc += (
            postp_coeff[:, dim + 1] * cc[dim] + postp_coeff[:, -1] * cc[dim] ** 2
        )

    # Compute norm
    error = np.sum((V * (postp_cc - true_pcc) ** 2) ** 0.5) / np.sum(
        (V * (true_pcc) ** 2) ** 0.5
    )

    return error





def l2_nv_conf_p(g, d, kw, true_nv):

    # First, rotate the grid
    g_rot = rotate_embedded_grid(g)

    # Obtain node values
    nv = g_rot.nodes

    # Retrieve conforming coefficients
    conf_coeff = d["error_estimates"]["sh"].copy()

    # Evaluate at the nodes
    nn = g.num_nodes
    nc = g.num_cells
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([nc, g.dim + 1]))
    nodes_coor_cell = np.empty([g.dim, nc, g.dim + 1])
    for dim in range(g.dim):
        nodes_coor_cell[dim] = nv[dim][nodes_cell]

    # Compute P1 reconstructed nodal pressures
    rec_pnv = matlib.repmat(conf_coeff[:, 0].reshape([nc, 1]), 1, g.dim + 1)
    for dim in range(g.dim):
        rec_pnv += (
            matlib.repmat(conf_coeff[:, dim + 1].reshape([nc, 1]), 1, g.dim + 1)
            * nodes_coor_cell[dim]
        )

    # Flattening and formating
    idx = np.array([np.where(nodes_cell.flatten() == x)[0][0] for x in np.arange(nn)])
    rec_pnv_flat = rec_pnv.flatten()
    recons_pnv = rec_pnv_flat[idx]

    # Compute error
    error = np.sum(((recons_pnv - true_nv) ** 2) ** 0.5) / np.sum(
        ((true_nv) ** 2) ** 0.5
    )

    return error





def l2_postp(g, d, idx_top, idx_middle, idx_bot):

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]

    # Retrieve postprocess pressure
    ph_coeff = d["error_estimates"]["ph"].copy()

    # Quadpyfying arrays
    if g.dim == 1:
        alpha = _quadpyfy(ph_coeff[:, 0], int_point)
        beta = _quadpyfy(ph_coeff[:, 1], int_point)
        epsilon = _quadpyfy(ph_coeff[:, 2], int_point)
    elif g.dim == 2:
        alpha = _quadpyfy(ph_coeff[:, 0], int_point)
        beta = _quadpyfy(ph_coeff[:, 1], int_point)
        gamma = _quadpyfy(ph_coeff[:, 2], int_point)
        epsilon = _quadpyfy(ph_coeff[:, 3], int_point)

    # Define fracture regions for the bulk
    def top_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5
        post = alpha + beta * x + gamma * y + epsilon * (x ** 2 + y ** 2)
        int_ = (p_ex - post) ** 2

        return int_

    def mid_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2) ** 0.5
        post = alpha + beta * x + gamma * y + epsilon * (x ** 2 + y ** 2)
        int_ = (p_ex - post) ** 2

        return int_

    def bot_subregion(X):
        x = X[0]
        y = X[1]

        p_ex = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
        post = alpha + beta * x + gamma * y + epsilon * (x ** 2 + y ** 2)
        int_ = (p_ex - post) ** 2

        return int_

    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        post = alpha + beta * x + epsilon * x ** 2

        int_ = (p_ex - post) ** 2

        return int_

    # Compute errors
    if g.dim == 2:

        int_top = method.integrate(top_subregion, elements)
        int_mid = method.integrate(mid_subregion, elements)
        int_bot = method.integrate(bot_subregion, elements)
        integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
    elif g.dim == 1:

        integral = method.integrate(fracture, elements)

    return integral.sum()





def l2_new_postpro(gb):

    for e, d_e in gb.edges():
        # Obtain mortar grid, mortar fluxes, adjacent grids, and data dicts
        g_m = d_e["mortar_grid"]
        g_l, g_h = gb.nodes_of_edge(e)
        d_h = gb.node_props(g_h)
        d_l = gb.node_props(g_l)

    lam = d_e[pp.STATE]["interface_flux"]
    # Declare integration method, obtain number of integration points,
    # normalize the integration points (quadpy uses -1 to 1), and obtain
    # weights, which have to be divided by two for the same reason
    num_points = 3
    normalized_intpts = np.array([0, 0.5, 1])

    # ------------- Treatment of the high-dimensional grid --------------------

    # Retrieve mapped faces from master to mortar
    face_high, _, _, = sps.find(g_m.mortar_to_master_avg())

    # Find, to which cells "face_high" belong to
    cell_faces_map, cell_idx, _ = sps.find(g_h.cell_faces)
    _, facehit, _ = np.intersect1d(cell_faces_map, face_high, return_indices=True)
    cell_high = cell_idx[facehit]  # these are the cells where face_high live

    # Obtain the coordinates of the nodes of each relevant face
    face_nodes_map, _, _ = sps.find(g_h.face_nodes)
    node_faces = face_nodes_map.reshape((np.array([g_h.num_faces, g_h.dim])))
    node_faces_high = node_faces[face_high]
    nodescoor_faces_high = np.zeros(
        [g_h.dim, node_faces_high.shape[0], node_faces_high.shape[1]]
    )
    for dim in range(g_h.dim):
        nodescoor_faces_high[dim] = g_h.nodes[dim][node_faces_high]

    # Reformat node coordinates to match size of integration point array
    nodecoor_format_high = np.empty([g_h.dim, face_high.size, num_points * g_h.dim])
    for dim in range(g_h.dim):
        nodecoor_format_high[dim] = matlib.repeat(
            nodescoor_faces_high[dim], num_points, axis=1
        )

    # Obtain evaluation points at the higher-dimensional faces
    faces_high_intcoor = np.empty([g_h.dim, face_high.size, num_points])
    for dim in range(g_h.dim):
        faces_high_intcoor[dim] = (
            nodecoor_format_high[dim][:, num_points:]
            - nodecoor_format_high[dim][:, :num_points]
        ) * normalized_intpts + nodecoor_format_high[dim][:, :num_points]

    # Retrieve postprocessed pressure
    ph = d_h["error_estimates"]["ph"].copy()
    ph_cell_high = ph[cell_high]

    # Evaluate postprocessed higher dimensional pressure at integration points
    tracep_intpts = _quadpyfy(ph_cell_high[:, 0], num_points)
    for dim in range(g_h.dim):
        tracep_intpts += (
            _quadpyfy(ph_cell_high[:, dim + 1], num_points) * faces_high_intcoor[dim]
            + _quadpyfy(ph_cell_high[:, -1], num_points) * faces_high_intcoor[dim] ** 2
        )

    # -------------- Treatment of the low-dimensional grid --------------------

    # First, rotate the grid
    gl_rot = rotate_embedded_grid(g_l)

    # Obtain indices of the cells, i.e., slave to mortar mapped cells
    cell_low, _, _, = sps.find(g_m.mortar_to_slave_avg())

    # Obtain n coordinates of the nodes of those cells
    cell_nodes_map, _, _ = sps.find(g_l.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([g_l.num_cells, g_l.dim + 1]))
    nodes_cell_low = nodes_cell[cell_low]
    nodescoor_cell_low = np.zeros(
        [g_l.dim, nodes_cell_low.shape[0], nodes_cell_low.shape[1]]
    )
    for dim in range(g_l.dim):
        nodescoor_cell_low[dim] = gl_rot.nodes[dim][nodes_cell_low]

    # Reformat node coordinates to match size of integration point array
    nodecoor_format_low = np.empty([g_l.dim, cell_low.size, num_points * (g_l.dim + 1)])
    for dim in range(g_l.dim):
        nodecoor_format_low[dim] = matlib.repeat(
            nodescoor_cell_low[dim], num_points, axis=1
        )

    # Obtain evaluation at the lower-dimensional cells
    cells_low_intcoor = np.empty([g_l.dim, cell_low.size, num_points])
    for dim in range(g_l.dim):
        cells_low_intcoor[dim] = (
            nodecoor_format_low[dim][:, num_points:]
            - nodecoor_format_low[dim][:, :num_points]
        ) * normalized_intpts + nodecoor_format_low[dim][:, :num_points]

    mortar_highdim_areas = g_m.master_to_mortar_avg() * g_h.face_areas
    lam_vel = lam / mortar_highdim_areas
    lam_vel_broad = _quadpyfy(lam_vel, num_points)

    # Obtain side grids and projection matrices
    side0, side1 = g_m.project_to_side_grids()

    proj_matrix_side0 = side0[0]
    proj_matrix_side1 = side1[0]

    p_low = -0.5 * (
        (proj_matrix_side0 * lam_vel_broad + proj_matrix_side1 * lam_vel_broad)
        - (proj_matrix_side0 * tracep_intpts + proj_matrix_side1 * tracep_intpts)
    )

    point_pressures = p_low
    point_coordinates = cells_low_intcoor[0][: g_l.num_cells]

    from error_estimates_reconstruction import _oswald_1d, P2_reconstruction

    ph_coeff = P2_reconstruction(g_l, point_pressures, point_coordinates)
    d_l["error_estimates"]["ph"] = ph_coeff.copy()

    # Declare integration parameters
    method = qp.line_segment.newton_cotes_closed(5)
    int_point = method.points.shape[0]

    # Retrieve elements
    elements = _get_quadpy_elements(g_l, gl_rot)

    # Apply Oswald interpolator
    point_p, point_coor = _oswald_1d(g_l, gl_rot, "flow", d_l)

    # Obain sh coeff
    sh_coeff = P2_reconstruction(g_l, point_p, point_coor)

    # Quadpyfying arrays
    c0 = _quadpyfy(sh_coeff[:, 0], int_point)
    c1 = _quadpyfy(sh_coeff[:, 1], int_point)
    c2 = _quadpyfy(sh_coeff[:, 2], int_point)

    # Define fracture regions for the bulk
    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        conf = c0 * x ** 2 + c1 * x + c2

        int_ = (p_ex - conf) ** 2

        return int_

    integral = method.integrate(fracture, elements)

    return np.sum(integral[: int(integral.size)])


def l2_sh(g, d):

    kw = "flow"
    sd_operator_name = "diffusion"
    p_name = "pressure"

    from error_estimates_utility import get_postp_coeff
    from error_estimates_reconstruction import _oswald_1d, P2_reconstruction

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    method = qp.line_segment.newton_cotes_closed(5)
    int_point = method.points.shape[0]

    # Get postprocessed pressures coefficients
    ph_coeff = get_postp_coeff(g, g_rot, d, kw, sd_operator_name, p_name)
    d["error_estimates"]["ph"] = ph_coeff.copy()

    # Apply Oswald interpolator
    point_p, point_coor = _oswald_1d(g, g_rot, kw, d)

    # Obain sh coeff
    sh_coeff = P2_reconstruction(g, point_p, point_coor)

    # Quadpyfying arrays
    c0 = _quadpyfy(sh_coeff[:, 0], int_point)
    c1 = _quadpyfy(sh_coeff[:, 1], int_point)
    c2 = _quadpyfy(sh_coeff[:, 2], int_point)

    # Define fracture regions for the bulk
    # Define integration regions for the fracture
    def fracture(X):
        x = X  # [0]

        p_ex = -1
        conf = c0 * x ** 2 + c1 * x + c2

        int_ = (p_ex - conf) ** 2

        return int_

    integral = method.integrate(fracture, elements)

    return np.sum(integral[: int(integral.size)])


# def l2_direct_recons(g, d, idx_top, idx_middle, idx_bot):

#     from error_estimates_utility import _compute_node_pressure_kavg
#     from error_estimates_reconstruction import _P1

#     # Rotate grid
#     g_rot = rotate_embedded_grid(g)

#     # Nodal values
#     p_nv = _compute_node_pressure_kavg(g, g_rot, d, "flow", "diffusion", "pressure")

#     # Conforming coefficients
#     sh_coeff = _P1(g, g_rot, p_nv)

#     # Retrieve elements
#     elements = _get_quadpy_elements(g, g_rot)

#     # Declaring integration methods
#     if g.dim == 1:
#         method = qp.line_segment.newton_cotes_closed(5)
#         int_point = method.points.shape[0]
#     elif g.dim == 2:
#         method = qp.triangle.strang_fix_cowper_05()
#         int_point = method.points.shape[0]

#     # Quadpyfying arrays
#     if g.dim == 1:
#         alpha = _quadpyfy(sh_coeff[:, 0], int_point)
#         beta = _quadpyfy(sh_coeff[:, 1], int_point)
#     elif g.dim == 2:
#         alpha = _quadpyfy(sh_coeff[:, 0], int_point)
#         beta = _quadpyfy(sh_coeff[:, 1], int_point)
#         gamma = _quadpyfy(sh_coeff[:, 2], int_point)

#     # Define fracture regions for the bulk
#     def top_subregion(X):
#         x = X[0]
#         y = X[1]

#         p_ex = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5
#         conf = alpha + beta * x + gamma * y
#         int_ = (p_ex - conf) ** 2

#         return int_

#     def mid_subregion(X):
#         x = X[0]
#         y = X[1]

#         p_ex = ((x - 0.5) ** 2) ** 0.5
#         conf = alpha + beta * x + gamma * y
#         int_ = (p_ex - conf) ** 2

#         return int_

#     def bot_subregion(X):
#         x = X[0]
#         y = X[1]

#         p_ex = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
#         conf = alpha + beta * x + gamma * y
#         int_ = (p_ex - conf) ** 2

#         return int_

#     # Define integration regions for the fracture
#     def fracture(X):
#         x = X  # [0]

#         p_ex = -1
#         conf = alpha + beta * x

#         int_ = (p_ex - conf) ** 2

#         return int_

#     # Compute errors
#     if g.dim == 2:

#         int_top = method.integrate(top_subregion, elements)
#         int_mid = method.integrate(mid_subregion, elements)
#         int_bot = method.integrate(bot_subregion, elements)
#         integral = int_top * idx_top + int_mid * idx_middle + int_bot * idx_bot
#     elif g.dim == 1:

#         integral = method.integrate(fracture, elements)

#     return integral.sum()


def l2_grad_sh(g, d):

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieve elements
    elements = _get_quadpy_elements(g, g_rot)

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(5)
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        int_point = method.points.shape[0]

    # Retrieve velocity coefficients
    s_coeffs = d["error_estimates"]["sh"].copy()

    # Quadpyfying arrays
    if g.dim == 1:
        beta = _quadpyfy(s_coeffs[:, 1], int_point)
    elif g.dim == 2:
        beta = _quadpyfy(s_coeffs[:, 1], int_point)
        gamma = _quadpyfy(s_coeffs[:, 2], int_point)

    # Define integration regions for 2D subdomain
    def bulk(X):

        grad_x = beta
        grad_y = gamma

        int_ = (grad_x + grad_y) ** 2

        return int_

    # Define integration regions for the fracture
    def fracture(X):

        grad_x = beta

        int_ = (grad_x) ** 2

        return int_

    # Compute errors
    if g.dim == 2:
        integral = method.integrate(bulk, elements)
    elif g.dim == 1:
        integral = method.integrate(fracture, elements)

    return integral.sum()