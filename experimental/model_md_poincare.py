import porepy as pp
import numpy as np
import sympy as sym
import quadpy as qp
import mdestimates as mde
import mdestimates.estimates_utils as utils
from mdestimates._velocity_reconstruction import _internal_source_term_contribution as mortar_jump


def make_constrained_mesh(h=0.1):
    """
    Creates unstructured mesh for a given target mesh size for the case of a
    single vertical fracture embedded in the domain

    Parameters
    ----------
    h : float, optional
        Target mesh size. The default is 0.1.

    Returns
    -------
    gb : PorePy Object
        Porepy grid bucket object.

    """

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network_2d = pp.fracture_importer.network_2d_from_csv("network.csv", domain=domain)

    # Target lengths
    target_h_bound = h
    target_h_fract = h
    mesh_args = {"mesh_size_bound": target_h_bound, "mesh_size_frac": target_h_fract}
    # Construct grid bucket
    gb = network_2d.mesh(mesh_args, constraints=[1, 2])

    return gb


def get_2d_boundary_indices(g):
    """
    Obtain list of boolean indices for the boundary faces

    Parameters
    ----------
    g : Grid
        2D Grid.

    Raises
    ------
    ValueError
        If the g.dim is different from 2.

    Returns
    -------
    idx : List
        Containing the boolean indices as NumPy boolean arrays.
    """

    # Check dimensionality of the grid
    if g.dim != 2:
        raise ValueError("Dimension should be 2")

    # Obtaining coordinates
    x = g.face_centers

    # Get boundary faces
    bcf = g.get_boundary_faces()

    # Obtaining the boolean indices of the three different regions
    bottom = x[1][bcf] < 0.25
    middle = (x[1][bcf] >= 0.25) & (x[1][bcf] <= 0.75)
    top = x[1][bcf] > 0.75

    # We want only the boundary faces, not all of them
    bottom_bc = bcf[bottom]
    middle_bc = bcf[middle]
    top_bc = bcf[top]

    # Create a list containing the different indices for exporting
    bound_idx_list = [bottom_bc, middle_bc, top_bc]

    return bound_idx_list


def get_2d_cell_indices(g):
    """
    Get indices of the three regions where the bulk is decomposed

    Parameters
    ----------
    g : Porepy grid
        2D grid

    Raises
    ------
    ValueError
        If the g.dim is different from 2.

    Returns
    -------
    bool_list : List
        List of numpy boolean arrays containing the different regions.
    bool_array : Numpy Array of size g.num_cells
        Array containing the labeling (1 to 3) of the different regions.
    """

    # Check grid dimension
    if g.dim != 2:
        raise ValueError("This function is meant for the 2D bulk")

    # Get cell center coordinates
    x = g.cell_centers

    # Obtaining the boolean indices of the three different regions
    bottom = x[1] < 0.25
    middle = (x[1] >= 0.25) & (x[1] <= 0.75)
    top = x[1] > 0.75

    # Create a list containing the different indices for exporting
    cell_idx_list = [bottom, middle, top]

    # It is useful to assign a label to them, so we con plot them in Paraview
    subregions = 1 * bottom + 2 * middle + 3 * top

    return cell_idx_list, subregions


def get_exact_2d_pressure(g):
    """
    Get exact pressures (symbolic, NumPy, cell-centered) as given by the
    analytical solution of Validation 5.3 from the paper

    Parameters
    ----------
    g : PorePy Grid
        2D grid

    Raises
    ------
    ValueError
        If the grid dimension is different from 2.

    Returns
    -------
    p2d_sym_list : List of length 3
        List of symbolic expressions.
    p2d_numpy_list : List of length 3
        List of Lambda functions as given by p2d_sym_list.
    p2d_cc : NumPy nd-Array of size g_2d.num_cells
        Cell-centered pressures

    """

    # Check grid dimensionality
    if g.dim != 2:
        raise ValueError("Dimension must be 2")

    # Get cell center coordinates and cell center boolean indices
    x2d_cc = g.cell_centers
    cell_idx_list, _ = get_2d_cell_indices(g)

    # Define symbolic symbols
    x, y = sym.symbols("x y")

    # Define the three-dimensional exact form for each subregion. See also the
    # Appendix of the paper.
    p2d_bottom_sym = sym.sin(x*sym.pi) * sym.cos(y*sym.pi)
    p2d_middle_sym = sym.sin(x*sym.pi) * sym.cos(y*sym.pi)
    p2d_top_sym = sym.sin(x*sym.pi) * sym.cos(y*sym.pi)


    # Create a list that contains all symbolic expressions
    p2d_sym_list = [p2d_bottom_sym, p2d_middle_sym, p2d_top_sym]

    # Convert to Numpy expressions
    p2d_numpy_list = []
    for p in p2d_sym_list:
        p2d_numpy_list.append(sym.lambdify((x, y), p, "numpy"))

    # Obtain cell-center pressures
    p2d_cc = np.zeros(g.num_cells)
    for (p, idx) in zip(p2d_numpy_list, cell_idx_list):
        p2d_cc += p(x2d_cc[0], x2d_cc[1]) * idx

    return p2d_sym_list, p2d_numpy_list, p2d_cc


def get_2d_boundary_values(g, bound_idx_list, p2d_numpy_list):
    """
    Get boundary values (satisfying the exact pressure field) of the 3D domain

    Parameters
    ----------
    g : PorePy grid
        2D grid
    bound_idx_list : List of length 3
        List containing the boolean NumPy nd-arrays (of length g_2d.num_faces)
    p2d_numpy_list : List of length 3
        List containing the exact pressures as NumPy expressions

    Raises
    ------
    ValueError
        If the grid dimension is different from 2.

    Returns
    -------
    bc_values : NumPy nd-array of size g_2d.num_faces
        Boundary values as given by the exact solution.

    """

    # Check if grid is two-dimensional
    if g.dim != 2:
        raise ValueError("Dimension should be 2")

    # Get face-center coordinates
    x = g.face_centers

    # Initialize boundary values array
    bc_values = np.zeros(g.num_faces)

    # Evaluate exact pressure at external boundary faces at each region
    for (p, idx) in zip(p2d_numpy_list, bound_idx_list):
        bc_values[idx] = p(x[0][idx], x[1][idx])

    return bc_values


def get_exact_2d_pressure_gradient(g, p2d_sym_list):
    """
    Get exact pressure gradient (symbolic, NumPy, and cell-centered) for the 2D domain

    Parameters
    ----------
    g : PorePy grid
        2D grid.
    p2d_sym_list : List of length 3
        Containing the symbolic exact pressures for each subregion.

    Raises
    ------
    ValueError
        If the grid dimension is different from 2.

    Returns
    -------
    gradp2d_sym_list : List of length 3
        Containing the symbolic exact pressure gradient expressions
    gradp2d_numpy_list : List of length 3
        Containing the exact NumPy expressions for the pressure gradient.
    gradp2d_cc : NumPy nd-array of size g_2d.num_cells
        Cell-centered evaluated exact pressure gradient.

    """

    # Check dimensionality of the grid
    if g.dim != 2:
        raise ValueError("Dimension must be 2")

    # Define symbolic symbols
    x, y = sym.symbols("x y")

    # Get cell center coordinates, and cell center boolean indices
    x2d_cc = g.cell_centers
    cell_idx_list, _ = get_2d_cell_indices(g)

    # Obtain gradient of the pressures
    gradp2d_sym_list = []
    for p in p2d_sym_list:
        gradp2d_sym_list.append([sym.diff(p, x), sym.diff(p, y)])

    # Convert to Numpy expressions
    gradp2d_numpy_list = []
    for gradp in gradp2d_sym_list:
        gradp2d_numpy_list.append(
            [
                sym.lambdify((x, y), gradp[0], "numpy"),
                sym.lambdify((x, y), gradp[1], "numpy"),
            ]
        )

    # Obtain cell-centered pressure gradients
    gradpx_cc = np.zeros(g.num_cells)
    gradpy_cc = np.zeros(g.num_cells)
    gradpz_cc = np.zeros(g.num_cells)
    for (gradp, idx) in zip(gradp2d_numpy_list, cell_idx_list):
        gradpx_cc += gradp[0](x2d_cc[0], x2d_cc[1]) * idx
        gradpy_cc += gradp[1](x2d_cc[0], x2d_cc[1]) * idx
    gradp2d_cc = np.array([gradpx_cc, gradpy_cc, gradpz_cc])

    return gradp2d_sym_list, gradp2d_numpy_list, gradp2d_cc


def get_exact_2d_velocity(g, gradp2d_list):
    """
    Get exact velocity (symbolic, NumPy, and cell-centered) for the 2D domain

    Parameters
    ----------
    g : PorePy grid
        2D grid.
    gradp2d_list : List of length 3
        Containing the symbolic exact pressure gradient.

    Raises
    ------
    ValueError
        If the grid dimension is different from 2.

    Returns
    -------
    u2d_sym_list : List of length 3
        Containing the exact symbolic expressions for the velocity.
    u2d_numpy_list : List of length 3
        Containing the exact NumPy expressions for the velocity.
    u2d_cc : NumPy nd-Array of size g_2d.num_cells
        Containing the cell-centered evaluated exact velocity.

    """

    # Check dimensionality of the grid
    if g.dim != 2:
        raise ValueError("Dimension must be 2")

    # Define symbolic symbols
    x, y = sym.symbols("x y")

    # Get cell center coordinates, and cell center boolean indices
    x2d_cc = g.cell_centers
    cell_idx_list, _ = get_2d_cell_indices(g)

    # Obtain velocities
    u2d_sym_list = []
    for gradp in gradp2d_list:
        u2d_sym_list.append([-gradp[0], -gradp[1]])

    # Convert to Numpy expressions
    u2d_numpy_list = []
    for u in u2d_sym_list:
        u2d_numpy_list.append(
            [
                sym.lambdify((x, y), u[0], "numpy"),
                sym.lambdify((x, y), u[1], "numpy"),
            ]
        )

    # Obtain cell-centered pressure gradients
    ux_cc = np.zeros(g.num_cells)
    uy_cc = np.zeros(g.num_cells)
    uz_cc = np.zeros(g.num_cells)
    for (u, idx) in zip(u2d_numpy_list, cell_idx_list):
        ux_cc += u[0](x2d_cc[0], x2d_cc[1]) * idx
        uy_cc += u[1](x2d_cc[0], x2d_cc[1]) * idx
    u2d_cc = np.array([ux_cc, uy_cc, uz_cc])

    return u2d_sym_list, u2d_numpy_list, u2d_cc


def get_exact_2d_source_term(g, u2d_sym_list):
    """
    Get exact source term (satisfying the mass conservation equation) for the
    2D domain.

    Parameters
    ----------
    g : PorePy grid
        2D grid.
    u2d_sym_vel : List of length 3
        Containing the exact symbolic velocities for each subregion.

    Raises
    ------
    ValueError
        If the dimensionality of the grid is different from 2

    Returns
    -------
    f2d_sym_list : List of length 3
        Containing the exact symbolic source term.
    f2d_numpy_list : List of length 3
        Containing the exact NumPy expressions for the source term.
    f2d_cc : NumPy nd-array of size g_2d.num_cells
        Exact cell-centered evaluated source terms.

    """

    # Check grid dimensionality
    if g.dim != 2:
        raise ValueError("Dimension must be 2")

    # Define symbolic symbols
    x, y = sym.symbols("x y")

    # Get cell center coordinates, and cell center boolean indices
    x2d_cc = g.cell_centers
    cell_idx_list, _ = get_2d_cell_indices(g)

    # Obtain source term
    f2d_sym_list = []
    for u in u2d_sym_list:
        f2d_sym_list.append((sym.diff(u[0], x) + sym.diff(u[1], y)))

    # Convert to Numpy expressions
    f2d_numpy_list = []
    for f in f2d_sym_list:
        f2d_numpy_list.append(sym.lambdify((x, y), f, "numpy"))

    # Obtain cell-center source terms
    f2d_cc = np.zeros(g.num_cells)
    for (f, idx) in zip(f2d_numpy_list, cell_idx_list):
        f2d_cc += f(x2d_cc[0], x2d_cc[1]) * idx

    return f2d_sym_list, f2d_numpy_list, f2d_cc


def integrate_source_2d(g, f2d_numpy_list, cell_idx_list):
    """
    Computes the exact integral of the source term for the 2D domain

    Parameters
    ----------
    g : PorePy grid
        2D grid.
    f2d_numpy_list : List of length 3
        Containing the exact NumPy expressions for the source term.
    cell_idx_list : List of length 3
        Containing the boolean indices for the cells of the 2D domain

    Returns
    -------
    integral : NumPy nd-Array of size g_2d.num_cells
        Integral of the source term

    """

    # Declare integration method and get hold of elements in QuadPy format
    int_method = qp.t2.get_good_scheme(8)  # a scheme of degree 3 should be enough
    elements = utils.get_quadpy_elements(g, g)

    # We now declare the different integrand regions
    integral = np.zeros(g.num_cells)
    for (f, idx) in zip(f2d_numpy_list, cell_idx_list):

        # Declare integrand
        def integrand(x):
            return f(x[0], x[1])

        # Integrate, and add the contribution of each subregion
        integral += int_method.integrate(integrand, elements) * idx

    return integral


def compute_residual_error_2d(g, d, estimates, f_numpy_list, cell_idx_list):

    # Retrieve reconstructed velocity
    recon_u = d[estimates.estimates_kw]["recon_u"].copy()

    # Obtain coefficients of the full flux and compute its divergence
    u = utils.poly2col(recon_u)
    div_u = 2 * u[0]

    # Declare integration method and get hold of elements in QuadPy format
    int_method = qp.t2.get_good_scheme(8)
    elements = utils.get_quadpy_elements(g, g)

    # Compute integral
    integral = np.zeros(g.num_cells)
    for (f, idx) in zip(f_numpy_list, cell_idx_list):
        # Declare integrand
        def integrand(x):
            return (f(x[0], x[1]) - div_u) ** 2
        # Integrate, and add the contribution of each subregion
        integral += (int_method.integrate(integrand, elements) * idx)

    return integral

def compute_residual_error(g, d, estimates):
    """
    Computes residual errors for each subdomain grid

    Parameters
    ----------
    g: Grid
    d: Data dictionary
    estimates: Estimates object

    Returns
    -------
    Residual error (squared) for each cell of the subdomain.
    """

    # Retrieve reconstructed velocity
    recon_u = d[estimates.estimates_kw]["recon_u"].copy()

    # Obtain coefficients of the full flux and compute its divergence
    u = utils.poly2col(recon_u)
    if g.dim == 2:
        div_u = 2 * u[0]
    elif g.dim == 1:
        div_u = u[0]

    # Obtain contribution from mortar jump to local mass conservation
    jump_in_mortars = (mortar_jump(estimates, g) / g.cell_volumes).reshape(g.num_cells, 1)

    # Declare integration method and get hold of elements in QuadPy format
    if g.dim == 2:
        int_method = qp.t2.get_good_scheme(8)  # since f is quadratic, we need at least order 4
        elements = utils.get_quadpy_elements(g, g)
    elif g.dim == 1:
        int_method = qp.c1.newton_cotes_closed(4)
        elements = utils.get_quadpy_elements(g, utils.rotate_embedded_grid(g))

    # We now declare the different integrand regions and compute the norms
    integral = np.zeros(g.num_cells)
    if g.dim == 2:
        for (f, idx) in zip(f2d_numpy_list, cell_idx_list):
            # Declare integrand
            def integrand(x):
                return (f(x[0], x[1]) - div_u + jump_in_mortars) ** 2
            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx
    elif g.dim == 1:
        # Declare integrand
        def integrand(x):
            f_1d = -2 * np.ones_like(x)
            return (f_1d - div_u + jump_in_mortars) ** 2
        integral = int_method.integrate(integrand, elements)

    # Finally, obtain residual error
    residual_error = const.flatten() * integral

    return residual_error