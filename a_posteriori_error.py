#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:50:42 2020

@author: jv
"""

import error_estimates_utility as utils
import error_estimates_reconstruction as recons
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

    
    NOTE: The data dictionary on each node and edge of the entire Grid Bucket
    will be updated with the key d["error_estimates"], which in turn, will have
    the "diffusive_error" and "nonconf_error" keys, i.e., the (cell-wise) error
    estimates.


    ----------------------- General algorithm overview -----------------------
    
    [1] Flux-related calculations
        
        1.1 Compute full flux for each node of the grid bucket, and store in 
            d["error_estimates"]["full_flux"]
        1.2 Perform reconstruction of the subdomain velocities using RT0 extension
            of normal fluxes, and store them in d["error_estimates"]["recons_vel"]
            
    [2] Pressure-related calculations
    
        2.1 Postprocess pressure solution by locally integrating the RT0 
            velocity field, and store them in d["error_estimates"]["ph"]
        2.2 Reconstruct postprocessed pressure by applying the Oswald interpolator
            to the Lagrangian nodes, and store them in d["error_estimates"]["sh"]
    
    [3] Error evaluation using QuadPy
    
        3.1 Compute error estimators for the entire grid bucket and store them 
        respectively as d["error_estimates"]["diffusive_error"] and 
        d["error_estimates"]["nonconf_error"].
    
    --------------------------------------------------------------------------
    """

    # -------------------------------- START  --------------------------------

    # Create the field "error_estimates" inside the data dicitionary for
    # each node and edge of the entire grid bucket
    utils.init_estimates_data_keyword(gb)

    # ------------------------------ BLOCK [1] -------------------------------

    # 1.1: Compute full flux
    utils.compute_full_flux(gb, keyword, sd_operator_name, p_name, lam_name)

    # 1.2: Reconstruct flux
    recons.reconstruct_velocity(gb, keyword, lam_name)

    # ------------------------------ BLOCK [2] -------------------------------

    # 2.1: Postprocess pressure solution
    recons.postprocess_pressure(gb, keyword, sd_operator_name, p_name)

    # 2.2: Reconstruct postprocessed pressure
    recons.reconstruct_pressure(gb, keyword)

    # ------------------------------ BLOCK [3] -------------------------------

    # 3.1: Evaluate errors for each subdomain and each interface
    evaluate.compute_error_estimates(gb, keyword, lam_name)

    # --------------------------------- END ----------------------------------

    return None
