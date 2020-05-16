#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:47:00 2020

@author: jv
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import matplotlib
matplotlib.use('qt5Agg')

from a_posteriori_error import estimate_error, compute_global_error, compute_subdomain_error

#%% Generate the grid
def make_unstructured_grid(h=0.1):
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
    
    pts = np.array([[0.5, 0.5], [0.25, 0.75]])
    # Connection between the points (that is, the fractures) are specified as a 2 x num_frac array
    connections = np.array([[0], [1]])
    # Set the domain to the unit square, specified as a dictionary
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    # Define a 2d fracture network
    # network_2d = pp.FractureNetwork2d(pts, connections, domain)
    network_2d = pp.FractureNetwork2d(pts, connections, domain)
    # Plot fracture
    #network_2d.plot()
    # Target lengths
    target_h_bound = h
    target_h_fract = h
    mesh_args = {"mesh_size_bound": target_h_bound, "mesh_size_frac": target_h_fract}
    # Construct grid bucket
    gb = network_2d.mesh(mesh_args)

    return gb


def make_cartesian_grid(N=10):
    
    fracture = np.array([[0.5,  0.5], [0.25, 0.75]])
    gb = pp.meshing.cart_grid([fracture], [N, N], physdims=[1, 1])
    
    return gb


def fracture_tip_patches(g_2d):
    
    _, top_tip_idx, _ = sps.find( (g_2d.nodes[0] == 0.5) & (g_2d.nodes[1] == 0.75) ) 
    _, bot_tip_idx, _ = sps.find( (g_2d.nodes[0] == 0.5) & (g_2d.nodes[1] == 0.25) ) 
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
        p_ex  = d[pp.STATE]["p_exact"]
        
        # if g.dim == 2:
        #     frac_tip_patch = fracture_tip_patches(g)
        #     p_num[frac_tip_patch] = p_ex[frac_tip_patch]
        
        e = np.sqrt(np.sum(V * np.abs(p_num - p_ex)**2)) / np.sqrt(np.sum(V * np.abs(p_ex)**2))
        d[pp.STATE]["true_error"] = e
     
    for e, d in gb.edges():
         
        lam_num = d[pp.STATE]["interface_flux"]
        lam_ex = gb.cell_volumes_mortar()
        V = gb.cell_volumes_mortar()
        
        error = np.sqrt(np.sum(V * np.abs(lam_num - lam_ex)**2)) / np.sqrt(np.sum(V * np.abs(lam_ex)**2))
        
        d[pp.STATE]["true_error"] = error
         
         
     
    return None

#%% Obtain grid bucket
#h = 0.0125 # mesh size
#gb = make_unstructured_grid(h);
N = 40
gb = make_cartesian_grid(N)

g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)[0]

d_2d = gb.node_props(g_2d)
d_1d = gb.node_props(g_1d)
d_e = gb.edge_props([g_1d, g_2d])

xc_2d = g_2d.cell_centers
xc_1d = g_1d.cell_centers

xf_2d = g_2d.face_centers
xf_1d = g_1d.face_centers

# # Mappings
# cell_faces_map, _, _ = sps.find(g_2d.cell_faces)
# cell_nodes_map, _, _ = sps.find(g_2d.cell_nodes())

# # Cell-wise arrays
# faces_cell = cell_faces_map.reshape(np.array([g_2d.num_cells, g_2d.dim + 1]))
# frac_pairs = g_2d.frac_pairs.flatten()
# frac_side  = []
# q_val_side = []
# for face in frac_pairs:
#     cell = np.where(faces_cell == face)[0]
#     print('Face ', face, 'corresponds to cell', int(cell))
#     if 0.5 - xc_2d[0][cell] > 0 :
#         frac_side.append('left')
#         q_val_side.append(1)
#     else:
#         frac_side.append('right')
#         q_val_side.append(-1)
    
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
p2d_hor_sym = sym.sqrt((x - 0.5)**2)  # 0.25 <= y <= 0.75
p2d_top_sym = sym.sqrt((x - 0.5)**2 + (y - 0.75)**2)  # y > 0.75
p2d_bot_sym = sym.sqrt((x - 0.5)**2 + (y - 0.25)**2)  # y < 0.25

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
p1d = -1

# Mortar fluxes
lambda_left = 1
lambda_right = 1

# Fracture velocity
q1d = 0

# Fracture source term
f1d = -(lambda_left + lambda_right)

# Lambdifying the expressions
p2d_hor = sym.lambdify((x,y), p2d_hor_sym, "numpy")
p2d_top = sym.lambdify((x,y), p2d_top_sym, "numpy")
p2d_bot = sym.lambdify((x,y), p2d_bot_sym, "numpy")

q2d_hor = sym.lambdify((x,y), q2d_hor_sym, "numpy")
q2d_top = sym.lambdify((x,y), q2d_top_sym, "numpy")
q2d_bot = sym.lambdify((x,y), q2d_bot_sym, "numpy")

f2d_hor = 0
f2d_top = sym.lambdify((x,y), f2d_top_sym, "numpy")
f2d_bot = sym.lambdify((x,y), f2d_bot_sym, "numpy")

# Exact cell-center pressures
pcc_2d_exact = (p2d_hor(xc_2d[0], xc_2d[1]) * idx_hor_cc 
                + p2d_top(xc_2d[0], xc_2d[1]) * idx_top_cc 
                + p2d_bot(xc_2d[0], xc_2d[1]) * idx_bot_cc)

pcc_1d_exact = p1d * np.ones(g_1d.num_cells)

# Exact source terms
f2d = (f2d_hor * idx_hor_cc
       + f2d_top(xc_2d[0], xc_2d[1]) * idx_top_cc
       + f2d_bot(xc_2d[0], xc_2d[1]) * idx_bot_cc) * g_2d.cell_volumes

f1d = f1d * g_1d.cell_volumes

# # Exact face-center fluxes
# q2d_hor_fc = q2d_hor(xf_2d[0], xf_2d[1])
# q2d_hor_fc[0][0][frac_pairs] = q_val_side # take care of division by zero
# Q_2d_hor = q2d_hor_fc[0][0] * g_2d.face_normals[0] + q2d_hor_fc[1][0] * g_2d.face_normals[1]

# q2d_top_fc = q2d_top(xf_2d[0], xf_2d[1])
# Q_2d_top = q2d_top_fc[0][0] * g_2d.face_normals[0] + q2d_top_fc[1][0] * g_2d.face_normals[1]

# q2d_bot_fc = q2d_bot(xf_2d[0], xf_2d[1])
# Q_2d_bot = q2d_bot_fc[0][0] * g_2d.face_normals[0] + q2d_bot_fc[1][0] * g_2d.face_normals[1]

# Q_2d_exact = (Q_2d_hor * idx_hor_fc + Q_2d_top * idx_top_fc + Q_2d_bot * idx_bot_fc )

# Q_1d_exact = np.zeros(g_1d.num_faces)


#%% Obtain numerical solution

# Parameter assignment
# If you want to a setup which targets transport or mechanics problem,
# rather use the keyword 'transport' or 'mechanics'.
# For other usage, you will need to populate the parameter dictionary manually.
parameter_keyword = "flow"

# Maximum dimension of grids represented in the grid bucket
max_dim = gb.dim_max()

# Loop over all grids in the GridBucket.
# The loop will return a grid, and a dictionary used to store various data
for g, d in gb:
   
    # Define BoundaryCondition object
    bc_faces = g.get_boundary_faces()
    bc_type = bc_faces.size * ["dir"]
    bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
    specified_parameters = {"bc": bc}

    # Also set the values - specified as vector of size g.num_faces
    bc_values = np.zeros(g.num_faces)
    if g.dim == max_dim:
        bc_values[hor_bc_faces] = p2d_hor(xf_2d[0][hor_bc_faces], xf_2d[1][hor_bc_faces])
        bc_values[top_bc_faces] = p2d_top(xf_2d[0][top_bc_faces], xf_2d[1][top_bc_faces])
        bc_values[bot_bc_faces] = p2d_bot(xf_2d[0][bot_bc_faces], xf_2d[1][bot_bc_faces])
    else:
        bc_values[bc_faces] = -1
    
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
# We will use a multi-point flux approximation method on all subdomains
# Note that we need to provide the same keyword that we used when setting the parameters on the subdomain.
# If we change to pp.Mpfa(keyword='foo'), the discretization will not get access to the parameters it needs
subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)
source_discretization = pp.ScalarSource(keyword=parameter_keyword)

# On all subdomains, variables are identified by a string.
# This need not be the same on all subdomains, even if the governing equations are the same,
# but here we go for the simple option.
subdomain_variable = "pressure"

# Identifier of the discretization operator for each term.
# In this case, this is seemingly too complex, but for, say, an advection-diffusion problem
# we would need two discretizaiton objects (advection, diffusion) and one keyword
# per operator
subdomain_operator_keyword = "diffusion"

# Specify discertization objects on the interfaces / edges
# Again give the parameter keyword, and also the discretizations used on the two neighboring
# subdomains (this is needed to discretize the coupling terms)
edge_discretization = pp.RobinCoupling(
    parameter_keyword, subdomain_discretization, subdomain_discretization
)
# edge_discretization = pp.FluxPressureContinuity(parameter_keyword, subdomain_discretization)
# Variable name for the interface variable
edge_variable = "interface_flux"
# ... and we need a name for the discretization opertaor for each coupling term
coupling_operator_keyword = "interface_diffusion"

# Loop over all subdomains in the GridBucket, assign parameters
# Note that the data is stored in sub-dictionaries
for g, d in gb:
    # Assign primary variables on this grid, compatible with the designated discretization scheme
    # In this case, the discretization has one degree of freedom per cell.
    # If we changed to a mixed finite element method, this could be {'cells': 1, "faces": 1}
    d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
    # Assign discretization operator for the variable.
    # If the discretization is composed of several terms, they can be assigned
    # by multiple entries in the inner dictionary, e.g.
    #  {operator_keyword_1: method_1, operator_keyword_2: method_2, ...}
    d[pp.DISCRETIZATION] = {
        subdomain_variable: {subdomain_operator_keyword: subdomain_discretization,
                             "source": source_discretization}
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

# Now, store the absolute difference 
d_2d[pp.STATE]["abs_error"] = np.abs(pcc_2d_exact - d_2d[pp.STATE]["pressure"])
d_1d[pp.STATE]["abs_error"] = np.abs(pcc_1d_exact - d_1d[pp.STATE]["pressure"])


# Compute Darcy Fluxes
pp.fvutils.compute_darcy_flux(gb, lam_name=edge_variable)

# Obtaining the errors
# The errors are stored in the dictionaries under pp.STATE
#estimate_error(gb, lam_name=edge_variable, nodal_method="flux-inverse")

#%% Write to vtk. Create a new exporter, to avoid interferring with the above grid illustration.
exporter = pp.Exporter(gb, "flow", folder_name="md_flow")
exporter.write_vtk()
# Note that we only export the subdomain variable, not the one on the edge
exporter.write_vtk([subdomain_variable])

#%% Get true error

compute_l2_errors(gb)
print('The error in the bulk is:', d_2d[pp.STATE]["true_error"])
print('The error in the fracture is:', d_1d[pp.STATE]["true_error"])
print('The error in the interface is:', d_e[pp.STATE]["true_error"])


#%% Get global error
# global_error = compute_global_error(gb)
# g_2d = gb.grids_of_dimension(2)[0]
# g_1d = gb.grids_of_dimension(1)[0]
# d_2d = gb.node_props(g_2d)
# d_1d = gb.node_props(g_1d)
# print('The global error is: ', compute_global_error(gb))
# print('The error in the 2d grid is: ', compute_subdomain_error(g_2d, d_2d))
# print('The error in the 1d grid is: ', compute_subdomain_error(g_1d, d_1d))

#%% Plotting
#pp.plot_grid(g_2d, info="nc", alpha=.1)
pp.save_img('result.pdf', g_2d, pcc_2d_exact, plot_2d=True) # exact solution
#pp.plot_grid(g_2d, d_2d[pp.STATE]['pressure'], plot_2d=True) # approximate solution
# pp.plot_grid(g_2d, d_2d[pp.STATE]["abs_error"], plot_2d=True # absolute value of the diff



