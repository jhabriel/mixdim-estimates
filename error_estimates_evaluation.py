#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:17:52 2020

@author: jv
"""

import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sps
import quadpy as qp
import porepy as pp

from error_estimates_utility import rotate_embedded_grid

#%% Error computation
def evaluate_error_estimates(gb, kw):
    """
    Computes the subdomain difussive flux error estimate for each node of 
    the grid bucket.

    Parameters
    ----------
    gb : PorePy object
        Grid bucket
    kw : Keyword 
        Name of the problem, i.e., 'flow'
        
    Returns
    -------
    None
   
    NOTE: The error estimates are stored in d["error_estimates"]["diffusive_error"]
    """

    for g, d in gb:
        
        if g.dim > 0: # error estimates measured in the primal form, are only valid for g.dim > 0
            
            # First, rotate the grid. Note that if g == gb.dim_max(), this has
            # no effect.
            g_rot = rotate_embedded_grid(g)        
            
            # Retrieve reconstructed pressure coefficients
            if "recons_p" in d["error_estimates"]:
                p_coeff = d["error_estimates"]["recons_p"]
            else:
                raise("Reconstructed pressure coefficients not found.")
                
            # Retrieve reconstructed velocity coefficients
            if "recons_vel" in d["error_estimates"]:
                vel_coeff = d["error_estimates"]["recons_vel"]
            else:
                raise("Reconstructed velocity coefficients not found.")
        
            # Obtain the error
            diffusive_error = _subdomain_error(g_rot, d, kw, p_coeff, vel_coeff)
            
            # Store in data dictionary
            d["error_estimates"]["diffusive_error"] = diffusive_error

    return 

#%% Subdomain error 
def _subdomain_error(g, d, kw, p_coeff, vel_coeff):
    """
    Computes the diffusive flux error for a given node of the grid bucket

    Parameters
    ----------
    g : PorePy Object
        PorePy << rotated >> grid object
    d : Dictionary
        Data dictionary.
    kw : Keyword
        Problem name, i.e., 'flow'
    p_coeff : NumPy array
        Reconstructed pressure coefficients.
    vel_coeff : NumPy array
        Reconstructed velocity coefficients.

    Returns
    -------
    diffusive_error : NumPy array (g.num_cells)
        Diffusive flux error estimates for the given grid
    
    """
    
    # Boolean variable to check if we have bubbles or not
    has_bubbles = p_coeff.shape[1] != (g.dim + 1)
    #print('Has Bubbles?: ', has_bubbles)
    
    # Get quadpy elements
    elements = _get_quadpy_elements(g)
    if g.dim == 1: # Needed to please quadpy format for line segment integration
        elements = elements.reshape(g.dim + 1, g.num_cells)
    
    # Get permeability values
    # TODO: For now, we assume constant permeability on each cell
    perm = d[pp.PARAMETERS][kw]["second_order_tensor"].values
    perm_cell = perm[0][0]
    
    # Scale reconstructed velocity values by 1/sqrt(k)
    vel_coeff /= np.sqrt(matlib.repmat(perm_cell, vel_coeff.shape[1], 1)).T
        
    # Scale reconstructed pressures by sqrt(k)
    p_coeff *= np.sqrt(matlib.repmat(perm_cell, p_coeff.shape[1], 1)).T

    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.chebyshev_gauss_2(3)
        # degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        # degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 3:
        method = qp.tetrahedron.yu_2()
        # degree = method.degree
        int_point = method.points.shape[0]
    else:
        pass
        
    # Reformating scaled velocity coefficients
    if g.dim == 1:
        a = matlib.repmat(vel_coeff[:, 0], int_point, 1).T
        b = matlib.repmat(vel_coeff[:, 1], int_point, 1).T
    elif g.dim == 2:
        a = matlib.repmat(vel_coeff[:, 0], int_point, 1).T
        b = matlib.repmat(vel_coeff[:, 1], int_point, 1).T
        c = matlib.repmat(vel_coeff[:, 2], int_point, 1).T
    elif g.dim == 3:
        a = matlib.repmat(vel_coeff[:, 0], int_point, 1).T
        b = matlib.repmat(vel_coeff[:, 1], int_point, 1).T
        c = matlib.repmat(vel_coeff[:, 2], int_point, 1).T
        d = matlib.repmat(vel_coeff[:, 3], int_point, 1).T
    else:
        pass
    
    # Reformating scaled gradient of reconstructed pressure for P1 elements
    if g.dim == 1:
        beta = matlib.repmat(p_coeff[:, 1], int_point, 1).T
    elif g.dim == 2:
        beta = matlib.repmat(p_coeff[:, 1], int_point, 1).T
        gamma = matlib.repmat(p_coeff[:, 2], int_point, 1).T
    elif g.dim == 3:
        beta = matlib.repmat(p_coeff[:, 1], int_point, 1).T
        gamma = matlib.repmat(p_coeff[:, 2], int_point, 1).T
        delta = matlib.repmat(p_coeff[:, 3], int_point, 1).T
    else:
        pass
    
    # If we have a P1.5 pressure reconstruction, include the bubbles
    epsilon = matlib.repmat(p_coeff[:, -1], int_point, 1).T
    
    # Obtain integration functions according to the dimensionality
    if g.dim == 1:  
        def integrand(X):
            x = X[0]
            
            scaled_vel_x = a * x + b
            
            scaled_gradp_x = beta #+ has_bubbles * epsilon # 
            
            int_x = (scaled_vel_x + scaled_gradp_x) ** 2
            
            return int_x
        
    elif g.dim == 2:  
        def integrand(X):
            x = X[0]
            y = X[1]
            
            scaled_vel_x = a * x + b 
            scaled_vel_y = a * y + c 
            
            scaled_gradp_x = beta  + has_bubbles * epsilon * y
            scaled_gradp_y = gamma + has_bubbles * epsilon * x
            
            int_x = (scaled_vel_x + scaled_gradp_x) ** 2
            int_y = (scaled_vel_y + scaled_gradp_y) ** 2
            
            return int_x + int_y
    
    elif g.dim == 3:  
        def integrand(X):
            x = X[0]
            y = X[1]
            z = X[2]
            
            scaled_vel_x = a * x + b 
            scaled_vel_y = a * y + c 
            scaled_vel_z = a * z + d 
            
            scaled_gradp_x = beta  + has_bubbles * epsilon * y * z
            scaled_gradp_y = gamma + has_bubbles * epsilon * x * z
            scaled_gradp_z = delta + has_bubbles * epsilon * y * x
            
            int_x = (scaled_vel_x + scaled_gradp_x) ** 2
            int_y = (scaled_vel_y + scaled_gradp_y) ** 2
            int_z = (scaled_vel_z + scaled_gradp_z) ** 2
            
            return int_x + int_y+ int_z
    else:
        pass
    
    # Compute the integration
    diffusive_error =  method.integrate(integrand, elements)
    
    return diffusive_error

#%% Quadpy related functions

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
