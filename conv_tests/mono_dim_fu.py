#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:45:34 2020

@author: jv
"""

import numpy as np
import numpy.matlib as matlib
import matplotlib

import quadpy as qp
import sympy as sym
import scipy.sparse as sps

import porepy as pp
import matplotlib.pyplot as plt


from a_posteriori_error import estimate_error, compute_global_error, compute_subdomain_error
from error_evaluation import _get_quadpy_elements
 
    
def convergence_mono_grid(mesh_size):
    
    def make_grid(mesh_size=0.1, domain_lenght=[1.0, 1.0]):
    
        domain = {"xmin": 0, "xmax": domain_lenght[0], "ymin": 0, "ymax": domain_lenght[1]}
        network_2d = pp.FractureNetwork2d(None, None, domain)
        target_h_bound = target_h_fracture = target_h_min = mesh_size
    
        mesh_args = {
            "mesh_size_bound": target_h_bound,
            "mesh_size_frac": target_h_fracture,
            "mesh_size_min": target_h_min,
        }
    
        gb = network_2d.mesh(mesh_args)
    
        return gb
    
    def get_analytical_functions():
    
        x, y = sym.symbols("x y")
        p = x * (1-x) * y * (1-y)
        dpdx = sym.diff(p, x)
        dpdy = sym.diff(p, y)
        gradient_p = sym.Array([dpdx, dpdy])
        q = -gradient_p
        dqdx = sym.diff(q[0], x)
        dqdy = sym.diff(q[1], y)
        div_q = dqdx + dqdy
        rhs = div_q
        p_ex = sym.lambdify((x, y), p, "numpy")
        q_ex = sym.lambdify((x, y), q, "numpy")
        rhs_ex = sym.lambdify((x, y), rhs, "numpy")
        gradP_ex = sym.lambdify((x, y), gradient_p, "numpy")
    
        return p_ex, q_ex, rhs_ex, gradP_ex
    
    # %% Parameter Assigment
    def assing_data(grid, data, parameter_keyword):
    
        g = grid
        d = data
        kw_f = parameter_keyword
    
        # Boundary condititions
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        labels = np.array(["dir"] * b_faces.size)
        bc = pp.BoundaryCondition(g, b_faces, labels)
        bc_values = np.zeros(g.num_faces)
    
        # Permeability (CURRRENTLY ONLY WORKING FOR UNIT PERMEABILITY)
        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx)
    
        # Getting source term
        _, _, rhs, _ = get_analytical_functions()
        source = rhs(g.cell_centers[0], g.cell_centers[1])
    
        # Create a dictionary to override the default parameters
        specified_parameters = {
            "second_order_tensor": perm,
            "bc": bc,
            "bc_values": bc_values,
            "source": source * g.cell_volumes,
        }
    
        # Initialize dictionary
        pp.initialize_default_data(g, d, kw_f, specified_parameters)
        
        return None
    
    # CREATING MESH
    #mesh_size = 0.005
    gb = make_grid(mesh_size, [1, 1])
    #gb = make_structured_grid([10, 10], [1, 1])
    g = gb.grids_of_dimension(2)[0]
    d = gb.node_props(g)
    #pp.save_img("grid.pdf", g, info="cfn", alpha=0.25, figsize=(20, 20))
    
    # RETRIEVING EXACT DATA
    exact_pressure, exact_darcy_velocity, exact_rhs, exact_gradP = get_analytical_functions()
    p_cc_ex = exact_pressure(g.cell_centers[0], g.cell_centers[1])
    # Exact pressure at the nodes
    p_nv_ex = exact_pressure(g.nodes[0], g.nodes[1])
    # Exact Darcy's velocity (vector field) at the faces centers
    q_fc_ex = exact_darcy_velocity(g.face_centers[0], g.face_centers[1])
    # Exact Darcy's flux (scalar field) at the faces centers
    #Q_fc_ex = compute_exact_fluxes(g)
    # Exact right hand side (source term)
    rhs_ex = exact_rhs(g.cell_centers[0], g.cell_centers[1])
    
    # PARAMETER ASSIGNMENT
    parameter_keyword = "flow"
    assing_data(g, d, parameter_keyword)
    
    # DEFINE VARIABLES, ASSEMBLE, DISCRETIZE
    subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)
    source_discretization = pp.ScalarSource(keyword=parameter_keyword)
    
    subdomain_variable = "pressure"
    subdomain_operator_keyword = "diffusion"
    d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
    d[pp.DISCRETIZATION] = {
        subdomain_variable: {
            subdomain_operator_keyword: subdomain_discretization,
            "scalar_source": source_discretization,
        }
    }
    assembler = pp.Assembler(gb)
    assembler.discretize()
    A, b = assembler.assemble_matrix_rhs()
    
    # SOLVE THE LINEAR SYSTEM
    p_cc = sps.linalg.spsolve(A, b)
    _ = pp.set_state(d, {subdomain_variable: p_cc})
    
    # EXTRACT NUMERICAL FLUXES
    pp.fvutils.compute_darcy_flux(gb, keyword=parameter_keyword)
    Q_fc = d[pp.PARAMETERS][parameter_keyword]["darcy_flux"]
    
    #%% Error estimation
    estimate_error(g, data=d)
    error_DF = d[pp.STATE]["error_DF"].sum()
    
    #%% True error
    x, y = sym.symbols("x y")
    p = x * (1-x) * y * (1-y)
    dpdx = sym.diff(p, x)
    dpdy = sym.diff(p, y)
    gradient_p = sym.Array([dpdx, dpdy])
    q = -gradient_p
    dqdx = sym.diff(q[0], x)
    dqdy = sym.diff(q[1], y)
    div_q = dqdx + dqdy
    rhs = div_q
    p_ex = sym.lambdify((x, y), p, "numpy")
    q_ex = sym.lambdify((x, y), q, "numpy")
    rhs_ex = sym.lambdify((x, y), rhs, "numpy")
    gradP_ex = sym.lambdify((x, y), gradient_p, "numpy")
    
    elements = _get_quadpy_elements(g)
    p_coeffs = d[pp.STATE]["p_coeff"]
    
    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.chebyshev_gauss_2(3)
        degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 2:
        method = qp.triangle.strang_fix_cowper_05()
        degree = method.degree
        int_point = method.points.shape[0]
    elif g.dim == 3:
        method = qp.tetrahedron.yu_2()
        degree = method.degree
        int_point = method.points.shape[0]
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
    
    def grad_pex_minus_pnum(X):
        
        x = X[0]
        y = X[1]  
        grad_pex_x = -x*y*(1 - y) + y*(1 - x)*(1 - y)
        grad_pex_y = -x*y*(1 - x) + x*(1 - x)*(1 - y)
        grad_pre_x = beta
        grad_pre_y = gamma
        
        int_x = (grad_pex_x - grad_pre_x)**2
        int_y = (grad_pex_y - grad_pre_y)**2
            
        return int_x + int_y
    
    int_ = method.integrate(grad_pex_minus_pnum, elements)
    true_primal_error = int_.sum()
    d[pp.STATE]["h_max"] = gb.diameter()
    d[pp.STATE]["true_primal_error"] = true_primal_error
    d[pp.STATE]["effectivity_idx"] =  d[pp.STATE]["error_DF"].sum() / d[pp.STATE]["true_primal_error"]
    #print('The error estimate is: ', d[pp.STATE]["error_DF"].sum())
    #print('The true estimate is: ', d[pp.STATE]["true_primal_error"])
    #print('The effectivity index is: ', d[pp.STATE]["effectivity_idx"])
    
    h_max = gb.diameter()
    true_error = int_.sum()
    error_DF = d[pp.STATE]["error_DF"].sum()
    eff_idx = d[pp.STATE]["effectivity_idx"]
    
    return h_max, true_error, error_DF, eff_idx
    
    
     