#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:51:32 2020

@author: jv
"""
import numpy as np
import numpy.matlib as matlib
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp

from a_posteriori_new import estimate_error
from error_estimates_utility import (
    rotate_embedded_grid,
    compute_global_error, 
    compute_subdomain_error, 
    compute_interface_error,
    transfer_error_to_state,
    _get_quadpy_elements,
    _quadpyfy,
)
        

def conv_fun_new(target_mesh_size=0.05, method="mpfa"):
    
    def make_constrained_mesh(h=0.05):
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
        network_2d = pp.fracture_importer.network_2d_from_csv(
            "conv_test_2d.csv", domain=domain
        )
        # Target lengths
        target_h_bound = h
        target_h_fract = h
        mesh_args = {
            "mesh_size_bound": target_h_bound,
            "mesh_size_frac": target_h_fract,
        }
        # Construct grid bucket
        gb = network_2d.mesh(mesh_args, constraints=[1, 2])

        return gb

    def fracture_tip_patches(g_2d):

        _, top_tip_idx, _ = sps.find((g_2d.nodes[0] == 0.5) & (g_2d.nodes[1] == 0.75))
        _, bot_tip_idx, _ = sps.find((g_2d.nodes[0] == 0.5) & (g_2d.nodes[1] == 0.25))
        cell_node_map = g_2d.cell_nodes()
        _, top_patch, _ = sps.find(cell_node_map[top_tip_idx[0]])
        _, bot_patch, _ = sps.find(cell_node_map[bot_tip_idx[0]])
        discard_cells = np.hstack([top_patch, bot_patch])

        return discard_cells

    def compute_l2_errors(gb):

        for g, d in gb:

            V = g.cell_volumes
            A = g.face_areas

            p_num = d[pp.STATE]["pressure"]
            p_ex = d[pp.STATE]["p_exact"]

            # if g.dim == 2:
            #     frac_tip_patch = fracture_tip_patches(g)
            #     p_num[frac_tip_patch] = p_ex[frac_tip_patch]

            e = np.sqrt(np.sum(V * np.abs(p_num - p_ex) ** 2)) / np.sqrt(
                np.sum(V * np.abs(p_ex) ** 2)
            )
            d[pp.STATE]["true_error"] = e

        for e, d in gb.edges():

            lam_num = d[pp.STATE]["interface_flux"]
            lam_ex = gb.cell_volumes_mortar()
            V = gb.cell_volumes_mortar()

            error = np.sqrt(np.sum(V * np.abs(lam_num - lam_ex) ** 2)) / np.sqrt(
                np.sum(V * np.abs(lam_ex) ** 2)
            )

            d[pp.STATE]["true_error"] = error

        return None

    def fv(scheme):
        if scheme == "tpfa" or scheme == "mpfa":
            return True
        elif scheme == "rt0" or scheme == "mvem":
            return False
        else:
            raise ("Method unrecognize")

    #%% Obtain grid bucket
    gb = make_constrained_mesh(target_mesh_size)

    g_2d = gb.grids_of_dimension(2)[0]
    g_1d = gb.grids_of_dimension(1)[0]

    h_max = gb.diameter()

    d_2d = gb.node_props(g_2d)
    d_1d = gb.node_props(g_1d)
    d_e = gb.edge_props([g_1d, g_2d])

    xc_2d = g_2d.cell_centers
    xc_1d = g_1d.cell_centers

    xf_2d = g_2d.face_centers
    xf_1d = g_1d.face_centers

    # Mappings
    cell_faces_map, _, _ = sps.find(g_2d.cell_faces)
    cell_nodes_map, _, _ = sps.find(g_2d.cell_nodes())

    # Cell-wise arrays
    faces_cell = cell_faces_map.reshape(np.array([g_2d.num_cells, g_2d.dim + 1]))
    frac_pairs = g_2d.frac_pairs.flatten()
    frac_side = []
    q_val_side = []
    for face in frac_pairs:
        cell = np.where(faces_cell == face)[0]
        # print('Face ', face, 'corresponds to cell', int(cell))
        if 0.5 - xc_2d[0][cell] > 0:
            frac_side.append("left")
            q_val_side.append(1)
        else:
            frac_side.append("right")
            q_val_side.append(-1)

    # Boolean indices of the cell centers
    idx_hor_cc = (xc_2d[1] >= 0.25) & (xc_2d[1] <= 0.75)
    idx_top_cc = xc_2d[1] > 0.75
    idx_bot_cc = xc_2d[1] < 0.25

    # Boolean indices of the face centers
    idx_hor_fc = (xf_2d[1] >= 0.25) & (xf_2d[1] <= 0.75)
    idx_top_fc = xf_2d[1] > 0.75
    idx_bot_fc = xf_2d[1] < 0.25

    # Indices of boundary faces
    bnd_faces = g_2d.get_boundary_faces()
    idx_hor_bc = (xf_2d[1][bnd_faces] >= 0.25) & (xf_2d[1][bnd_faces] <= 0.75)
    idx_top_bc = xf_2d[1][bnd_faces] > 0.75
    idx_bot_bc = xf_2d[1][bnd_faces] < 0.25
    hor_bc_faces = bnd_faces[idx_hor_bc]
    top_bc_faces = bnd_faces[idx_top_bc]
    bot_bc_faces = bnd_faces[idx_bot_bc]

    #%% Obtain analytical solution
    x, y = sym.symbols("x y")

    # Bulk pressures
    p2d_hor_sym = sym.sqrt((x - 0.5) ** 2)  + 1 # 0.25 <= y <= 0.75
    p2d_top_sym = sym.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2)  + 1 # y > 0.75
    p2d_bot_sym = sym.sqrt((x - 0.5) ** 2 + (y - 0.25) ** 2)  + 1 # y < 0.25

    # Derivatives of the bulk pressure
    dp2d_hor_sym_dx = sym.diff(p2d_hor_sym, x)
    dp2d_hor_sym_dy = sym.diff(p2d_hor_sym, y)

    dp2d_top_sym_dx = sym.diff(p2d_top_sym, x)
    dp2d_top_sym_dy = sym.diff(p2d_top_sym, y)

    dp2d_bot_sym_dx = sym.diff(p2d_bot_sym, x)
    dp2d_bot_sym_dy = sym.diff(p2d_bot_sym, y)

    # Bulk velocities
    q2d_hor_sym = sym.Matrix([-dp2d_hor_sym_dx, -dp2d_hor_sym_dy])
    q2d_top_sym = sym.Matrix([-dp2d_top_sym_dx, -dp2d_top_sym_dy])
    q2d_bot_sym = sym.Matrix([-dp2d_bot_sym_dx, -dp2d_bot_sym_dy])

    # Bulk source terms
    f2d_hor_sym = 0
    f2d_top_sym = sym.diff(q2d_top_sym[0], x) + sym.diff(q2d_top_sym[1], y)
    f2d_bot_sym = sym.diff(q2d_bot_sym[0], x) + sym.diff(q2d_bot_sym[1], y)

    # Fracture pressure
    p1d = 0

    # Mortar fluxes
    lambda_left = 1
    lambda_right = 1

    # Fracture velocity
    q1d = 0

    # Fracture source term
    f1d = -(lambda_left + lambda_right)

    # Lambdifying the expressions
    p2d_hor = sym.lambdify((x, y), p2d_hor_sym, "numpy")
    p2d_top = sym.lambdify((x, y), p2d_top_sym, "numpy")
    p2d_bot = sym.lambdify((x, y), p2d_bot_sym, "numpy")

    q2d_hor = sym.lambdify((x, y), q2d_hor_sym, "numpy")
    q2d_top = sym.lambdify((x, y), q2d_top_sym, "numpy")
    q2d_bot = sym.lambdify((x, y), q2d_bot_sym, "numpy")

    f2d_hor = 0
    f2d_top = sym.lambdify((x, y), f2d_top_sym, "numpy")
    f2d_bot = sym.lambdify((x, y), f2d_bot_sym, "numpy")

    # Exact cell-center pressures
    pcc_2d_exact = (
        p2d_hor(xc_2d[0], xc_2d[1]) * idx_hor_cc
        + p2d_top(xc_2d[0], xc_2d[1]) * idx_top_cc
        + p2d_bot(xc_2d[0], xc_2d[1]) * idx_bot_cc
    )

    pcc_1d_exact = p1d * np.ones(g_1d.num_cells)

    # Exact source terms
    f2d = (
        f2d_hor * idx_hor_cc
        + f2d_top(xc_2d[0], xc_2d[1]) * idx_top_cc
        + f2d_bot(xc_2d[0], xc_2d[1]) * idx_bot_cc
    ) * g_2d.cell_volumes

    f1d = f1d * g_1d.cell_volumes

    # Exact face-center fluxes
    q2d_hor_fc = q2d_hor(xf_2d[0], xf_2d[1])
    q2d_hor_fc[0][0][frac_pairs] = q_val_side  # take care of division by zero
    Q_2d_hor = (
        q2d_hor_fc[0][0] * g_2d.face_normals[0]
        + q2d_hor_fc[1][0] * g_2d.face_normals[1]
    )

    q2d_top_fc = q2d_top(xf_2d[0], xf_2d[1])
    Q_2d_top = (
        q2d_top_fc[0][0] * g_2d.face_normals[0]
        + q2d_top_fc[1][0] * g_2d.face_normals[1]
    )

    q2d_bot_fc = q2d_bot(xf_2d[0], xf_2d[1])
    Q_2d_bot = (
        q2d_bot_fc[0][0] * g_2d.face_normals[0]
        + q2d_bot_fc[1][0] * g_2d.face_normals[1]
    )

    Q_2d_exact = Q_2d_hor * idx_hor_fc + Q_2d_top * idx_top_fc + Q_2d_bot * idx_bot_fc

    Q_1d_exact = np.zeros(g_1d.num_faces)

    #%% Obtain numerical solution

    parameter_keyword = "flow"
    max_dim = gb.dim_max()

    # Set parameters in the subdomains
    for g, d in gb:

        # Define BoundaryCondition object
        if g.dim == 2:
            bc_faces = g.get_boundary_faces()
        else:
            bc_faces = g.get_all_boundary_faces()
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        specified_parameters = {"bc": bc}

        # Also set the values - specified as vector of size g.num_faces
        bc_values = np.zeros(g.num_faces)
        if g.dim == max_dim:
            bc_values[hor_bc_faces] = p2d_hor(
                xf_2d[0][hor_bc_faces], xf_2d[1][hor_bc_faces]
            )
            bc_values[top_bc_faces] = p2d_top(
                xf_2d[0][top_bc_faces], xf_2d[1][top_bc_faces]
            )
            bc_values[bot_bc_faces] = p2d_bot(
                xf_2d[0][bot_bc_faces], xf_2d[1][bot_bc_faces]
            )
        else:
            bc_values[bc_faces] = 0

        specified_parameters["bc_values"] = bc_values

        # Source terms are given by the exact solution
        if g.dim == max_dim:
            source_term = f2d
        else:
            source_term = f1d

        specified_parameters["source"] = source_term

        # By using the method initialize_default_data, various other fields are also
        # added, see
        pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

    # Next loop over the edges (interfaces) in the
    for e, d in gb.edges():
        # On edges in the GridBucket, there is currently no methods for default initialization.

        # Set the normal diffusivity parameter (the permeability-like transfer coefficient)
        data = {"normal_diffusivity": 1}

        # Add parameters: We again use keywords to identify sets of parameters.
        #
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, parameter_keyword, data)

    #%% Primary variables and discretizations

    if method == "mpfa":
        subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)
    elif method == "tpfa":
        subdomain_discretization = pp.Tpfa(keyword=parameter_keyword)
    elif method == "rt0":
        subdomain_discretization = pp.RT0(keyword=parameter_keyword)
    elif method == "mvem":
        subdomain_discretization = pp.MVEM(keyword=parameter_keyword)
    else:
        raise ("Method not implemented")

    # Source term discretization
    if fv(method):
        source_discretization = pp.ScalarSource(keyword=parameter_keyword)
    else:
        source_discretization = pp.DualScalarSource(keyword=parameter_keyword)

    subdomain_variable = "pressure"
    subdomain_operator_keyword = "diffusion"
    edge_discretization = pp.RobinCoupling(
        parameter_keyword, subdomain_discretization, subdomain_discretization
    )
    edge_variable = "interface_flux"
    coupling_operator_keyword = "interface_diffusion"

    # Loop over all subdomains in the GridBucket, assign parameters
    # Note that the data is stored in sub-dictionaries
    if fv(method):
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
            d[pp.DISCRETIZATION] = {
                subdomain_variable: {
                    subdomain_operator_keyword: subdomain_discretization,
                    "source": source_discretization,
                }
            }
    else:
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {
                subdomain_variable: {
                    subdomain_operator_keyword: subdomain_discretization,
                    "source": source_discretization,
                }
            }

    # Next, loop over the edges
    for e, d in gb.edges():
        # Get the grids of the neighboring subdomains
        # The first will always be the lower-dimensional
        g1, g2 = gb.nodes_of_edge(e)
        # The interface variable has one degree of freedom per cell in the mortar grid
        # This is essentially a DG(0) discretization
        d[pp.PRIMARY_VARIABLES] = {edge_variable: {"cells": 1}}

        # The coupling discretization links an edge discretization with variables
        # and discretization operators on each neighboring grid
        d[pp.COUPLING_DISCRETIZATION] = {
            # This assignment associate this coupling term with a unique combination of variable
            # and term (thus discretization object) applied to each of the neighboring subdomains.
            # Again, the complexity is not warranted for this problem, but necessary for general
            # multi-physics problems with non-trivial couplings between variables.
            coupling_operator_keyword: {
                g1: (subdomain_variable, subdomain_operator_keyword),
                g2: (subdomain_variable, subdomain_operator_keyword),
                e: (edge_variable, edge_discretization),
            }
        }

    #%% Assembly and solve
    # Initialize the assembler with the GridBucket it should operate on.
    # Initialization will process the variables defined above (name, number, and number of dofs)
    # and assign global degrees of freedom.
    assembler = pp.Assembler(gb)

    # First discretize. This will call all discretizaiton operations defined above
    assembler.discretize()

    # Finally, assemble matrix and right hand side
    A, b = assembler.assemble_matrix_rhs()

    # Direct solver
    sol = sps.linalg.spsolve(A, b)

    # The solution vector is a global vector. Distribute it to the local grids and interfaces
    assembler.distribute_variable(sol)

    #%% Post-processing

    # First, store the exact solution in the data dictionary
    d_2d[pp.STATE]["p_exact"] = pcc_2d_exact
    d_1d[pp.STATE]["p_exact"] = pcc_1d_exact

    # if not fv(method):
    #     d_2d[pp.STATE]["pressure"] = subdomain_discretization.extract_pressure(
    #         g_2d, d_2d[pp.STATE]["pressure"], d_2d
    #     )
    #     d_1d[pp.STATE]["pressure"] = subdomain_discretization.extract_pressure(
    #         g_1d, d_1d[pp.STATE]["pressure"], d_1d
    #     )

    # if fv(method):
    #     pp.plot_grid(g_2d, d_2d[pp.STATE]["pressure"], plot_2d=True)
    # else:
    #     #p = subdomain_discretization.extract_pressure(g_2d, d_2d[pp.STATE]["pressure"], d_2d)
    #     pp.plot_grid(g_2d, d_2d[pp.STATE]["pressure"], plot_2d=True)

    # Now, store the absolute difference
    # d_2d[pp.STATE]["abs_error"] = np.abs(pcc_2d_exact - d_2d[pp.STATE]["pressure"])
    # d_1d[pp.STATE]["abs_error"] = np.abs(pcc_1d_exact - d_1d[pp.STATE]["pressure"])

    # Compute Darcy Fluxes
    # pp.fvutils.compute_darcy_flux(gb, lam_name=edge_variable)

    # Obtaining the errors
    # The errors are stored in the dictionaries under pp.STATE
    estimate_error(gb, lam_name=edge_variable)
    
    # Transfer results to the d[pp.STATE]
    transfer_error_to_state(gb)
    
    
    #%% Compute errors

    def compute_true_errors(gb):

        # Compute errors for the bulk and the fracture
        for g, d in gb:

            # Rotate grid
            g_rot = rotate_embedded_grid(g)

            # Retrieving quadpy elemnts
            elements = _get_quadpy_elements(g, g_rot)
            
            # Retrieve pressure coefficients
            p_coeffs = d["error_estimates"]["postprocess_p"].copy()
            
            # Declaring integration methods
            if g.dim == 1:
                method = qp.line_segment.newton_cotes_closed(5)
                int_point = method.points.shape[0]
            elif g.dim == 2:
                method = qp.triangle.strang_fix_cowper_05()
                int_point = method.points.shape[0]

            # Coefficients of the gradient of postprocessed pressure
            if g.dim == 1:
                beta = _quadpyfy(p_coeffs[:, 1], int_point)
                epsilon = _quadpyfy(p_coeffs[:, -1], int_point)
            elif g.dim == 2:
                beta = _quadpyfy(p_coeffs[:, 1], int_point)
                gamma = _quadpyfy(p_coeffs[:, 2], int_point)
                epsilon = _quadpyfy(p_coeffs[:, -1], int_point)

            # Define integration regions for 2D subdomain
            def top_subregion(X):
                x = X[0]
                y = X[1]
                
                grad_pex_x = (x - 0.5) / ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** (0.5)
                grad_pex_y = (y - 0.75) / ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** (0.5)
                grad_pre_x = beta + 2 * epsilon * x
                grad_pre_y = gamma + 2 * epsilon * y

                int_x = (grad_pex_x - grad_pre_x) ** 2
                int_y = (grad_pex_y - grad_pre_y) ** 2

                return int_x + int_y

            def mid_subregion(X):
                x = X[0]
                y = X[1]
                
                grad_pex_x = ((x - 0.5) ** 2) ** (0.5) / (x - 0.5)
                grad_pex_y = 0
                grad_pre_x = beta + 2 * epsilon * x
                grad_pre_y = gamma + 2 * epsilon * y

                int_x = (grad_pex_x - grad_pre_x) ** 2
                int_y = (grad_pex_y - grad_pre_y) ** 2

                return int_x + int_y

            def bot_subregion(X):
                x = X[0]
                y = X[1]
                
                grad_pex_x = (x - 0.5) / ((x - 0.5) ** 2 + (y - 0.25) ** 2 ) ** (0.5)
                grad_pex_y = (y - 0.25) / ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** (0.5)
                grad_pre_x = beta + 2 * epsilon * x
                grad_pre_y = gamma + 2 * epsilon * y

                int_x = (grad_pex_x - grad_pre_x) ** 2
                int_y = (grad_pex_y - grad_pre_y) ** 2

                return int_x + int_y

            # Define integration regions for the fracture
            def fracture(X):
                x = X[0]
                
                grad_pre_x = beta + 2 * epsilon * x
                
                int_x = (-grad_pre_x) ** 2

                return int_x

            # Compute errors
            if g.dim == 2:

                int_top = method.integrate(top_subregion, elements)
                int_mid = method.integrate(mid_subregion, elements)
                int_bot = method.integrate(bot_subregion, elements)
                integral = (
                    int_top * idx_top_cc + int_mid * idx_hor_cc + int_bot * idx_bot_cc
                )
                d[pp.STATE]["true_error"] = integral

                num_cells_2d = g.num_cells
                true_error_2d = d[pp.STATE]["true_error"].sum()
                error_estimate_2d = compute_subdomain_error(g, d_2d)

            else:

                integral = method.integrate(fracture, elements)
                d[pp.STATE]["true_error"] = integral

                num_cells_1d = g.num_cells
                true_error_1d = d[pp.STATE]["true_error"].sum()
                error_estimate_1d = compute_subdomain_error(g, d_1d)
                
        
        # Compute true error for the interface
        for e, d in gb.edges():
            
            g_m = d_e["mortar_grid"]
            lam = d_e[pp.STATE][edge_variable]
            
            g_l, g_h = gb.nodes_of_edge(e)
            d_h = gb.node_props(g_h)
            d_l = gb.node_props(g_l)
            
            
            # --- Project high-dim face-center reconst pressure to the mortar grid ---
            
            # First, rotate the grid (whenever coordinates are used, the rotated
            # version of the grids must be used).
            gh_rot = rotate_embedded_grid(g_h)
            
            # Obtain indices of the fracture faces
            _, frac_faces, _ = sps.find(g_h.tags["fracture_faces"])
            
            # Obtain face-center coordinates of the fracture faces.
            # NOTE: Rows are the fracture faces, and cols are the spatial coordinates
            fc_h = gh_rot.face_centers[:, frac_faces].T
            
            # Obtain global (subdomain) face-cell mapping and associated cell indices
            face_of_cell_map, cell_idx , _   = sps.find(g_h.cell_faces)
            
            # We need the indices where face_of_cell_map matches frac_faces 
            _, face_matched_idx, _ = np.intersect1d(face_of_cell_map, frac_faces, return_indices=True)
            
            # Retrieve the cells associated with the fracture faces
            cells_h = cell_idx[face_matched_idx]
            
            # Now, retrieve the coefficients of the reconstructed pressure of those cells
            ph_coeff = d_h["error_estimates"]["postprocess_p"][cells_h]
            
            # Obtain the contiguous face-center pressure to the interface.
            ph_fc = ph_coeff[:, 0]
            for dim in range(g_h.dim):
                ph_fc += ph_coeff[:, dim+1] * fc_h[:, dim] + ph_coeff[:, -1] * fc_h[:, dim] ** 2
            
            # Prepare to project
            ph_faces = np.zeros(g_h.num_faces)
            ph_faces[frac_faces] = ph_fc
            
            # Project to the mortar grid
            proj_fc_ph = g_m.master_to_mortar_avg() * ph_faces
            
            
            # --- Project low-dim cell-center reconst pressure to the mortar grid ---
        
            # First, rotate the grid (whenever coordinates are used, the rotated
            # version of the grids must be used).
            gl_rot = rotate_embedded_grid(g_l)
        
            # Rotate reconstructed pressure 
            pl_coeff = d_l["error_estimates"]["postprocess_p"]
        
            # Retrieve cell-center coordinates
            cc_l = gl_rot.cell_centers
            
            # Obtain reconstructed cell-center pressures
            pl_cc = pl_coeff[:, 0]
            for dim in range(g_l.dim):
                pl_cc += pl_coeff[:, dim+1] * cc_l[dim] + pl_coeff[:, -1] * cc_l[dim] ** 2
            
            # Project to the mortar grid    
            proj_cc_pl = g_m.slave_to_mortar_avg() * pl_cc     
            
            # -------------------- Estimate the diffusive error ----------------------
            
            # Compute the (mortar volume) scaled pressure difference.
            #diff = (proj_cc_pl - proj_fc_ph) * g_m.cell_volumes
            deltap_recon = (proj_cc_pl - proj_fc_ph)
            deltap_exact = -1 - 0
            mismatch_squared = (deltap_exact - deltap_recon) ** 2
            
            # Compute the mismatch
            true_error_mortar = (mismatch_squared * g_m.cell_volumes).sum()
            
            
            # -------------------- Projection to the side grids ----------------------
            
            # Obtain side grids and projection matrices
            side0, side1 = g_m.project_to_side_grids()
            
            # proj_matrix_side0 = side0[0]
            # grid_side0 = side0[1]
            # diff_side0 = proj_matrix_side0 * diffusive_error
            
            # proj_matrix_side1 = side1[0]
            # grid_side1 = side1[1]
            # diff_side1 = proj_matrix_side1 * diffusive_error

            # Compute true error on the interface: p_low - p_high
            num_cells_mortar = g_m.cell_volumes.size
            error_estimate_mortar = d_e[pp.STATE]["diffusive_error"].sum()
            
            
        return (
            num_cells_2d,
            true_error_2d,
            error_estimate_2d,
            num_cells_1d,
            true_error_1d,
            error_estimate_1d,
            num_cells_mortar,
            true_error_mortar, 
            error_estimate_mortar
        )

    # Retrieve errors
    (
        num_cells_2d,
        true_error_2d,
        error_estimate_2d,
        num_cells_1d,
        true_error_1d,
        error_estimate_1d,
        num_cells_mortar,
        true_error_mortar, 
        error_estimate_mortar
    ) = compute_true_errors(gb)

    # Return 
    return (
            num_cells_2d,
            true_error_2d,
            error_estimate_2d,
            num_cells_1d,
            true_error_1d,
            error_estimate_1d,
            num_cells_mortar,
            true_error_mortar, 
            error_estimate_mortar
        )
    
