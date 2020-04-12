#%% Importing modules
import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import quadpy as qp
import porepy as pp


#%% Computation of the diffusive flux error
def diffusive_flux_sd(grid, data, pressure_coeffs, velocity_coeffs):
    """
    Computes the subdomain's diffusive flux error

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object.
    data : dictionary 
        Dictionary containing the parameters.
    pressure_coeffs : NumPy array
        Pressure coefficients for each element of the grid.
    velocity_coeffs : NumPy array
        Velocity coefficients for each element of the grid.
    quadpy_elements : NumPy array
        Elements in QuadPy format.
        
    Returns
    -------
    None
   
    """

    # Get quadpy elements
    quadpy_elements = _get_quadpy_elements(grid)

    # Obtaining the error according to the reconstruction degree
    if pressure_coeffs.shape[1] == grid.dim + 1:
        _diffusive_flux_error_sd_p1(
            grid, data, pressure_coeffs, velocity_coeffs, quadpy_elements
        )
    else:
        _diffusive_flux_error_sd_p12(
            grid, data, pressure_coeffs, velocity_coeffs, quadpy_elements
        )

    return None


#%% Computation of the diffusive flux error for P1,2 elements
def _diffusive_flux_error_sd_p12(
    grid, data, pressure_coeffs, velocity_coeffs, quadpy_elements
):
    """
    Computes the subdomain's diffusive flux error for P1,2 pressure reconstruction

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object.
    data : dictionary 
        Dictionary containing the parameters.
    pressure_coeffs : NumPy array
        Pressure coefficients for each element of the grid.
    velocity_coeffs : NumPy array
        Velocity coefficients for each element of the grid.
    quadpy_elements : NumPy array
        Elements in QuadPy format.
        
    Returns
    -------
    None

    """

    # Renaming variables
    g = grid
    p_coeffs = pressure_coeffs
    q_coeffs = velocity_coeffs
    elements = quadpy_elements

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.chebyshev_gauss_2(3)
        degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_03()
        degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 3:
        method = qp.tetrahedron.yu_2()
        degree = method.degree
        int_point = method.points.shape[0]
    else:
        pass

    # Coefficients of the reconstructed velocity field
    if g.dim == 1:
        a = matlib.repmat(q_coeffs[:, 0], int_point, 1).T
        b = matlib.repmat(q_coeffs[:, 1], int_point, 1).T
    elif g.dim == 2:
        a = matlib.repmat(q_coeffs[:, 0], int_point, 1).T
        b = matlib.repmat(q_coeffs[:, 1], int_point, 1).T
        c = matlib.repmat(q_coeffs[:, 2], int_point, 1).T
    elif g.dim == 3:
        a = matlib.repmat(q_coeffs[:, 0], int_point, 1).T
        b = matlib.repmat(q_coeffs[:, 1], int_point, 1).T
        c = matlib.repmat(q_coeffs[:, 2], int_point, 1).T
        d = matlib.repmat(q_coeffs[:, 3], int_point, 1).T
    else:
        pass

    # Coefficients of the gradient of reconstructed pressure for P1,2 elements
    if g.dim == 1:
        beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
        epsilon = matlib.repmat(p_coeffs[:, -1], int_point, 1).T
    elif g.dim == 2:
        beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
        gamma = matlib.repmat(p_coeffs[:, 2], int_point, 1).T
        epsilon = matlib.repmat(p_coeffs[:, -1], int_point, 1).T
    elif g.dim == 3:
        beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
        gamma = matlib.repmat(p_coeffs[:, 2], int_point, 1).T
        delta = matlib.repmat(p_coeffs[:, 3], int_point, 1).T
        epsilon = matlib.repmat(p_coeffs[:, -1], int_point, 1).T
    else:
        pass

    # Integrand #1: velocity \cdot velocity
    def vel_dot_vel(x):

        if g.dim == 1:
            int_ = (a * x[0] + b) ** 2
        elif g.dim == 2:
            int_ = (a * x[0] + b) ** 2 + (a * x[1] + c) ** 2
        elif g.dim == 3:
            int_ = (a * x[0] + b) ** 2 + (a * x[1] + c) ** 2 + (a * x[2] + d) ** 2
        else:
            int_ = None

        return int_

    # Integrand #2: velocity \cdot grad(pressure)
    def vel_dot_gradP(x):

        if g.dim == 1:
            int_ = (a * x[0] + b) * (beta + 2 * epsilon * x[0])
        elif g.dim == 2:
            int_ = (a * x[0] + b) * (beta + 2 * epsilon * x[0]) + (a * x[1] + c) * (
                gamma + 2 * epsilon * x[1]
            )
        elif g.dim == 3:
            int_ = (
                (a * x[0] + b) * (beta + 2 * epsilon * x[0])
                + (a * x[1] + c) * (gamma + 2 * epsilon * x[1])
                + (a * x[2] + d) * (delta + 2 * epsilon * x[2])
            )
        else:
            int_ = None

        return int_

    # Integrand #3: grad(pressure) \cdot grad(pressure)
    def gradP_dot_gradP(x):

        if g.dim == 1:
            int_ = (beta + 2 * epsilon * x[0]) ** 2
        elif g.dim == 2:
            int_ = (beta + 2 * epsilon * x[0]) ** 2 + (gamma + 2 * epsilon * x[1]) ** 2
        elif g.dim == 3:
            int_ = (
                (beta + 2 * epsilon * x[0]) ** 2
                + (gamma + 2 * epsilon * x[1]) ** 2
                + (delta + 2 * epsilon * x[2]) ** 2
            )
        else:
            int_ = None

        return int_

    # Performing integrations
    int_1 = method.integrate(vel_dot_vel, elements)
    int_2 = method.integrate(vel_dot_gradP, elements)
    int_3 = method.integrate(gradP_dot_gradP, elements)

    # Computing diffusive flux error
    eta_DF = np.abs(int_1 + 2 * int_2 + int_3).flatten()

    # Create field in the data dictionary and store the error
    data[pp.STATE]["error_DF"] = eta_DF

    return None


#%% Computation of the diffusive flux error for P1 elements
def _diffusive_flux_error_sd_p1(
    grid, data, pressure_coeffs, velocity_coeffs, quadpy_elements
):
    """
    Computes the subdomain's diffusive flux error for P1 pressure reconstruction

    Parameters
    ----------
    grid : PorePy object
        Porepy grid object.
    data : dictionary 
        Dictionary containing the parameters.
    pressure_coeffs : NumPy array
        Pressure coefficients for each element of the grid.
    velocity_coeffs : NumPy array
        Velocity coefficients for each element of the grid.
    quadpy_elements : NumPy array
        Elements in QuadPy format.
        
    Returns
    -------
    None

    """

    # Renaming variables
    g = grid
    p_coeffs = pressure_coeffs
    q_coeffs = velocity_coeffs
    elements = quadpy_elements

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.chebyshev_gauss_2(3)
        degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_03()
        degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 3:
        method = qp.tetrahedron.yu_2()
        degree = method.degree
        int_point = method.points.shape[0]
    else:
        pass

    # Coefficients of the reconstructed velocity field
    if g.dim == 1:
        a = matlib.repmat(q_coeffs[:, 0], int_point, 1).T
        b = matlib.repmat(q_coeffs[:, 1], int_point, 1).T
    elif g.dim == 2:
        a = matlib.repmat(q_coeffs[:, 0], int_point, 1).T
        b = matlib.repmat(q_coeffs[:, 1], int_point, 1).T
        c = matlib.repmat(q_coeffs[:, 2], int_point, 1).T
    elif g.dim == 3:
        a = matlib.repmat(q_coeffs[:, 0], int_point, 1).T
        b = matlib.repmat(q_coeffs[:, 1], int_point, 1).T
        c = matlib.repmat(q_coeffs[:, 2], int_point, 1).T
        d = matlib.repmat(q_coeffs[:, 3], int_point, 1).T
    else:
        pass

    # Coefficients of the gradient of reconstructed pressure for P1 elements
    if g.dim == 1:
        beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
    elif g.dim == 2:
        beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
        gamma = matlib.repmat(p_coeffs[:, 2], int_point, 1).T
    elif g.dim == 3:
        beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
        gamma = matlib.repmat(p_coeffs[:, 2], int_point, 1).T
        delta = matlib.repmat(p_coeffs[:, 3], int_point, 1).T
    else:
        pass

    # Integrand #1: velocity \cdot velocity
    def vel_dot_vel(x):

        if g.dim == 1:
            int_ = (a * x[0] + b) ** 2
        elif g.dim == 2:
            int_ = (a * x[0] + b) ** 2 + (a * x[1] + c) ** 2
        elif g.dim == 3:
            int_ = (a * x[0] + b) ** 2 + (a * x[1] + c) ** 2 + (a * x[2] + d) ** 2
        else:
            int_ = None

        return int_

    # Integrand #2: velocity \cdot grad(pressure)
    def vel_dot_gradP(x):

        if g.dim == 1:
            int_ = a * beta * x[0] + b * beta
        elif g.dim == 2:
            int_ = (a * beta * x[0] + b * beta) + (a * gamma * x[1] + c * gamma)
        elif g.dim == 3:
            int_ = (
                (a * beta * x[0] + b * beta)
                + (a * gamma * x[1] + c * gamma)
                + (a * delta * x[2] + d * delta)
            )
        else:
            int_ = None

        return int_

    # Integrand #3: grad(pressure) \cdot grad(pressure)
    def gradP_dot_gradP(x):

        if g.dim == 1:
            int_ = beta ** 2
        elif g.dim == 2:
            int_ = beta ** 2 + gamma ** 2
        elif g.dim == 3:
            int_ = beta ** 2 + gamma ** 2 + delta ** 2
        else:
            int_ = None

        return int_

    # Performing integrations
    int_1 = method.integrate(vel_dot_vel, elements)
    int_2 = method.integrate(vel_dot_gradP, elements)
    int_3 = method.integrate(gradP_dot_gradP, elements)

    # Computing diffusive flux error
    eta_DF = np.abs(int_1 + 2 * int_2 + int_3).flatten()

    # Create field in the data dictionary and store the error
    data[pp.STATE]["error_DF"] = eta_DF

    return None


#%% Get quadpy elements
def _get_quadpy_elements(grid):
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
    g = grid
    nc = g.num_cells

    # Getting node coordinates for each cell
    cell_nodes_map, _, _ = sps.find(g.cell_nodes())
    nodes_cell = cell_nodes_map.reshape(np.array([nc, g.dim + 1]))
    nodes_coor_cell = np.empty([g.dim, nodes_cell.shape[0], nodes_cell.shape[1]])
    for dim in range(g.dim):
        nodes_coor_cell[dim] = g.nodes[dim][nodes_cell]

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

    return elements
