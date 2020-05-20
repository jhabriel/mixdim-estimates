import porepy as pp
from porepy.grids.grid_bucket import GridBucket

import pressure_reconstruction
import flux_reconstruction
import error_evaluation

import error_estimates_utility as util
import error_estimates_reconstruction as reconstruct
import error_estimates_evaluation as evaluate

def estimate_error(
    gb,
    keyword="flow",
    sd_operator_name="diffusion",
    p_name="pressure",
    lam_name="mortar_solution",
    nodal_method="flux-inverse",
    p_order="1",
    data=None,
):
    """
    Estimates the error of a mixed-dimensional elliptic problem. For the mono-dimensional
    case, the data dicitionary is a mandatory input field.
    
    Parameters
    ----------
    gb : PorePy object
        PorePy grid bucket object. Alternatively, g for mono-dimensional grids.
    keyword : keyword, optional
        Name of the problem. The default is "flow".
    sd_operator_name : Keyword, optional
        Subdomain operator name. The default is "diffusion"
    p_name : keyword, optional
        Name of the subdomain variable. The default is "pressure".
    lam_name : keyword, optional
        Name of the edge variable. The default is "mortar_solution".
    nodal_method : keyword, optional
        Name of the nodal pressure reconstruction method. The default is 'flux-inverse'. 
        The other implemented option is 'k-averaging'.
    p_order : keyword, optional
        Order of pressure reconstruction. The default is "1", i.e. P1 elements. 
        The other implement option is '1.5', which refers to P1 elements enriched
        with purely parabolic terms.
    data : dictionary, optional
        Data dictionary. The default is None. 
    Returns
    -------
    None.

    """

    # Compute errors for mono-dimensional grids
    if not isinstance(gb, GridBucket) and not isinstance(gb, pp.GridBucket):
        _estimate_error_mono(gb, data, keyword, p_name, nodal_method, p_order)

    # Compute errors for mixed-dimensional grids
    else:
        _estimate_error_mixed(
            gb, keyword, sd_operator_name, p_name, lam_name, nodal_method, p_order
        )

    return None


# -------------------------------------------------------------------------- #
#                Error estimation for mixed-dimensional grids                #
# -------------------------------------------------------------------------- #
def _estimate_error_mixed(
    gb, kw, sd_operator_name, p_name, lam_name, nodal_method, p_order
):

    """
    ----------------------- General algorithm overview -----------------------
    
    [1] Compute necessary quantities, and store in the data dicitionaries
        
        1.1 Compute full flux for each node of the grid bucket, and store in 
            d["error_estimates"]["full_flux"]
        1.2 Compute node pressure for each node of the grid bucket, and store 
            in d["error_estimates"]["node_pressure"]
        1.3 Compute node mortar flux for each edge of the grid bucket, and store
            in d["error_estimates"]["nodal_mortar_flux"]
            
    [2] Perform reconstructions
    
        2.1 Perform reconstruction of the subdomain velocities using RT0 extension
            of normal fluxes, and store in d["error_estimates"]["recons_vel"]
        2.2 Perform reconstruction of the subdomain pressures using P1 or 
            P1 + bubbles elements, and store in d["error_esimates"]["recons_p"]
        2.3 Perform reconstruction of the interfaces mortar fluxes using P1
            elements, and store in d["error_estimates"]["recons_lambda"]
    
    [3] Evaluate errors using QuadPy
    
        3.1 Compute diffusive flux error for the entire grid bucket (i.e, nodes
            and edges), and and store in d["error_estimates"]["diffusive_flux_error"]
    
    --------------------------------------------------------------------------
    """
    
    # First, we create the field ["error_estimates"] inside the data dictionary
    # of each node and edge of the grid bucket
    util.init_estimates_data_keyword(gb)
        
    # ------------------------------ BLOCK [1] -------------------------------
    
    # 1.1: Compute full flux
    util.compute_full_flux(gb, kw, sd_operator_name, p_name, lam_name)
    
    # 1.2: Compute node pressure
    util.compute_node_pressure(gb, kw, sd_operator_name, p_name, nodal_method)
    
    # TODO: 1.3: Compute node mortar fluxes 
    #util.compute_node_mortar_flux()

    # ------------------------------ BLOCK [2] -------------------------------
    
    # 2.1: Reconstruct subdomain velocities
    reconstruct.subdomain_velocity(gb, kw, lam_name)

    # 2.2: Reconstruct subdomain pressures
    reconstruct.subdomain_pressure(gb, sd_operator_name, p_name, p_order)

    # TODO: 2.3: Reconstruct interface mortar fluxes
    #reconstruct.interface_mortar_flux(gb, kw)    

    # ------------------------------ BLOCK [3] -------------------------------
    
    # 3.1: Evaluate errors
    evaluate.evaluate_error_estimates(gb, kw)
    
    #flux_reconstruction.compute_full_flux(gb, kw, sd_operator_name, p_name, lam_name)

    # # Compute the errors on the subdomains by looping through the nodes
    # for g, d in gb:

    #     if (
    #         g.dim > 0
    #     ):  # errors measured in the primal norm only defined for g.dim > 0 (?)

    #         # Rotate all grids to reference coordinate system
    #         g_rot = _rotate_grid(g)

    #         # Get reconstructed velocity coefficients for the grid g
    #         v_coeffs = flux_reconstruction.subdomain_velocity(
    #             gb, g, g_rot, d, kw, lam_name
    #         )

    #         # Get reconstructed pressure coefficients for the grid g
    #         p_coeffs = pressure_reconstruction.subdomain_pressure(
    #             gb,
    #             g_rot,
    #             d,
    #             kw,
    #             sd_operator_name,
    #             p_name,
    #             lam_name,
    #             nodal_method,
    #             p_order,
    #         )

    #         # Evaluate errors with quadpy and store results in d[pp.STATE]
    #         error_evaluation.diffusive_flux_sd(g_rot, d, p_coeffs, v_coeffs)

    # # Compute the errors on the interfaces by looping through the edges
    # for e, d in gb.edges():

    #     # # Retrieve neighboring grids and mortar grid
    #     # g_low, g_high = self.gb.nodes_of_edge(e)
    #     # mortar_grid = d["mortar_grid"]

    #     # # We assume that the lower dimensional grid coincides geometrically
    #     # # with the mortar grid. Hence, we rotate the g_low
    #     # glow_rot = self.rotate(g_low)

    #     # # Get reconstructed interface fluxes
    #     # lambda_coeffs = flux_reconstruction.interface_flux(glow_rot, mortar_grid, d, self.kw)

    #     # # TODO: Calculate errors in the interfaces
    #     pass


# -------------------------------------------------------------------------- #
#                 Error estimation for mono-dimensional grids                #
# -------------------------------------------------------------------------- #
def _estimate_error_mono(g, sd_discr, d, kw, p_name, nodal_method, p_order):
    """
    Estimates the error of a mono-dimensional grid

    Parameters
    ----------
    g : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.
    kw : TYPE
        DESCRIPTION.
    p_name : TYPE
        DESCRIPTION.
    nodal_method : TYPE
        DESCRIPTION.
    p_order : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # Perform pressure reconstruction
    p_coeffs = pressure_reconstruction.mono_grid_pressure(
        g, d, sd_discr, kw, p_name, nodal_method, p_order
    )

    # Perform flux reconstruction
    v_coeffs, _ = flux_reconstruction.mono_grid_velocity(g, d, sd_discr, kw)

    # Evaluate errors by means of numerical integration and store in d[pp.STATE]
    error_evaluation.diffusive_flux_sd(g, d, p_coeffs, v_coeffs)

    return None


# -------------------------------------------------------------------------- #
#                             Utility functions                              #
# -------------------------------------------------------------------------- #
def _rotate_grid(g):
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
    g_rot : Porepy object
        Rotated PorePy grid.
        
    """

    # Copy grid to keep original one untouched
    g_rot = g.copy()

    # Rotate grid
    (
        cell_centers,
        face_normals,
        face_centers,
        R,
        dim,
        nodes,
    ) = pp.map_geometry.map_grid(g_rot)

    # Update rotated fields in the relevant dimension
    for dim in range(g.dim):
        g_rot.cell_centers[dim] = cell_centers[dim]
        g_rot.face_normals[dim] = face_normals[dim]
        g_rot.face_centers[dim] = face_centers[dim]
        g_rot.nodes[dim] = nodes[dim]

    return g_rot


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

    # Obtain global error for mono-dimensional grid
    if not isinstance(gb, GridBucket) and not isinstance(gb, pp.GridBucket):
        global_error = data["error_estimates"]["difussive_error"].sum()
    
    # Obtain global error for mixed-dimensional grids
    else:
        for g, d in gb:
            if g.dim > 0:
                global_error += d["error_estimates"]["diffusive_error"].sum()

        for e, d in gb.edges():
            # TODO: add diffusive flux error contribution for the interfaces
            # global_error += d[]
            pass

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

    subdomain_error = d["error_estimates"]["diffusive_error"].sum()

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

    return interface_error
