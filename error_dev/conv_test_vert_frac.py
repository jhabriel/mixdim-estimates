import numpy as np
import numpy.matlib as matlib
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import matplotlib
#matplotlib.use('qt5Agg')


#from a_posteriori_error import estimate_error
from a_posteriori_new import estimate_error
from error_estimates_utility import (
    rotate_embedded_grid, 
    transfer_error_to_state, 
    compute_global_error,
    compute_subdomain_error,
    _quadpyfy,
    _get_quadpy_elements,
)
#from grid_refinement import refine_mesh_by_splitting

from error_estimates_reconstruction import _oswald_1d

#%% Generate the grid
def make_constrained_mesh(h=0.05):
    
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network_2d = pp.fracture_importer.network_2d_from_csv('conv_test_2d.csv', domain=domain)
    # Target lengths
    target_h_bound = h
    target_h_fract = h
    mesh_args = {"mesh_size_bound": target_h_bound, "mesh_size_frac": target_h_fract}
    # Construct grid bucket
    gb = network_2d.mesh(mesh_args, constraints=[1, 2])
    
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
        
    return None

#%% Obtain grid bucket
h = 0.1# mesh size
gb = make_constrained_mesh(h)

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
frac_side  = []
q_val_side = []
for face in frac_pairs:
    cell = np.where(faces_cell == face)[0]
    #print('Face ', face, 'corresponds to cell', int(cell))
    if 0.5 - xc_2d[0][cell] > 0 :
        frac_side.append('left')
        q_val_side.append(1)
    else:
        frac_side.append('right')
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

# Exact face-center fluxes
q2d_hor_fc = q2d_hor(xf_2d[0], xf_2d[1])
q2d_hor_fc[0][0][frac_pairs] = q_val_side # take care of division by zero
Q_2d_hor = q2d_hor_fc[0][0] * g_2d.face_normals[0] + q2d_hor_fc[1][0] * g_2d.face_normals[1]

q2d_top_fc = q2d_top(xf_2d[0], xf_2d[1])
Q_2d_top = q2d_top_fc[0][0] * g_2d.face_normals[0] + q2d_top_fc[1][0] * g_2d.face_normals[1]

q2d_bot_fc = q2d_bot(xf_2d[0], xf_2d[1])
Q_2d_bot = q2d_bot_fc[0][0] * g_2d.face_normals[0] + q2d_bot_fc[1][0] * g_2d.face_normals[1]

Q_2d_exact = (Q_2d_hor * idx_hor_fc + Q_2d_top * idx_top_fc + Q_2d_bot * idx_bot_fc )

Q_1d_exact = np.zeros(g_1d.num_faces)


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
    if g.dim == 2:
        bc_faces = g.get_boundary_faces()
    else:
        bc_faces = g.get_all_boundary_faces()
    
    #bc_faces = g.get_boundary_faces()    
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
    # The interface variable one degree of freedom per cell in the mortar grid
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
#pp.fvutils.compute_darcy_flux(gb, lam_name=edge_variable)

# Obtaining the errors
# The errors are stored in the dictionaries under pp.STATE
#estimate_error(gb, lam_name=edge_variable, nodal_method="k-averaging", p_order="1")
estimate_error(gb, lam_name=edge_variable)


#%% Write to vtk. Create a new exporter, to avoid interferring with the above grid illustration.
exporter = pp.Exporter(gb, "flow", folder_name="md_flow")
exporter.write_vtk()
exporter.write_vtk("pressure")
# Note that we only export the subdomain variable, not the one on the edge

#%% Get true error

#compute_l2_errors(gb)
#print('The error in the bulk is:', d_2d[pp.STATE]["true_error"])
#print('The error in the fracture is:', d_1d[pp.STATE]["true_error"])

#%% Get global errora
global_error = compute_global_error(gb)
g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)[0]
d_2d = gb.node_props(g_2d)
d_1d = gb.node_props(g_1d)
print('The global error is: ', global_error)
print('The error in the 2d grid is: ', compute_subdomain_error(g_2d, d_2d))
print('The error in the 1d grid is: ', compute_subdomain_error(g_1d, d_1d))
# 
#%% Plotting
#pp.plot_grid(g_2d, info="nc", alpha=.1)
#pp.plot_grid(g_2d, pcc_2d_exact, plot_2d=True) # exact solution
pp.plot_grid(g_2d, d_2d[pp.STATE]['pressure'], plot_2d=True) # approximate solution
# pp.plot_grid(g_2d, d_2d[pp.STATE]["abs_error"], plot_2d=True # absolute value of the diff

# #%% Preparing to measure the exact error

# elements = _get_quadpy_elements(g_2d)
# p_coeffs = d_2d["error_estimates"]["recons_p"]
# g = g_2d

# # Declaring integration methods
# if g.dim == 1:
#     method = qp.line_segment.chebyshev_gauss_2(3)
#     degree = method.degree
#     int_point = method.points.shape[0]
# elif g.dim == 2:
#     method = qp.triangle.strang_fix_cowper_05()
#     degree = method.degree
#     int_point = method.points.shape[0]
# elif g.dim == 3:
#     method = qp.tetrahedron.yu_2()
#     degree = method.degree
#     int_point = method.points.shape[0]
# else:
#     pass

# # Coefficients of the gradient of reconstructed pressure for P1 elements
# if g.dim == 1:
#     beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
# elif g.dim == 2:
#     beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
#     gamma = matlib.repmat(p_coeffs[:, 2], int_point, 1).T
# elif g.dim == 3:
#     beta = matlib.repmat(p_coeffs[:, 1], int_point, 1).T
#     gamma = matlib.repmat(p_coeffs[:, 2], int_point, 1).T
#     delta = matlib.repmat(p_coeffs[:, 3], int_point, 1).T
# else:
#     pass

# def top_subregion(x):
    
#     grad_pex_x = (x[0]-0.5) / ((x[0]-0.5)**2 + (x[1]-0.75)**2)**(0.5)
#     grad_pex_y = (x[1]-0.75) / ((x[0]-0.5)**2 + (x[1]-0.75)**2)**(0.5)
#     grad_pre_x = beta
#     grad_pre_y = gamma
    
#     int_x = (grad_pex_x - grad_pre_x)**2
#     int_y = (grad_pex_y - grad_pre_y)**2
    
#     return int_x + int_y

# def mid_subregion(x):
    
#     grad_pex_x = ((x[0]-0.5)**2)**(0.5) / (x[0]-0.5)
#     grad_pex_y = 0
#     grad_pre_x = beta
#     grad_pre_y = gamma
    
#     int_x = (grad_pex_x - grad_pre_x)**2
#     int_y = (grad_pex_y - grad_pre_y)**2
    
#     return int_x + int_y
     
# def bot_subregion(x):
    
#     grad_pex_x = (x[0]-0.5) / ((x[0]-0.5)**2 + (x[1]-0.25)**2)**(0.5)
#     grad_pex_y = (x[1]-0.25) / ((x[0]-0.5)**2 + (x[1]-0.25)**2)**(0.5)
#     grad_pre_x = beta
#     grad_pre_y = gamma
    
#     int_x = (grad_pex_x - grad_pre_x)**2
#     int_y = (grad_pex_y - grad_pre_y)**2
    
#     return int_x + int_y
    
# # Performing integrations
# int_top = method.integrate(top_subregion, elements)
# int_mid = method.integrate(mid_subregion, elements)
# int_bot = method.integrate(bot_subregion, elements)
# int_final = (int_top * idx_top_cc + int_mid * idx_hor_cc + int_bot * idx_bot_cc)
# int_final_sum = int_final.sum()
# print('The true error is: ', int_final_sum)

# d_2d[pp.STATE]["true_error"] = int_final
# d_1d[pp.STATE]["true_error"] = np.zeros(g_1d.num_cells)

# for g, d in gb:
#     d[pp.STATE]["diffusive_error"] = d["error_estimates"]["diffusive_error"]


# d_2d[pp.STATE]["efficiency_idx"] =  d_2d[pp.STATE]["diffusive_error"] / d_2d[pp.STATE]["true_error"]
# d_1d[pp.STATE]["efficiency_idx"] =  d_1d[pp.STATE]["diffusive_error"] * 0

# exporter.write_vtk([subdomain_variable, "diffusive_error", "true_error", "efficiency_idx"])

#%% Obtaining difference in pressure 
#pp.plot_grid(g_2d, info="cf", alpha=.1)
transfer_error_to_state(gb)

   
# Compute errors for the bulk and the fracture
for g, d in gb:

    # Rotate grid
    g_rot = rotate_embedded_grid(g)

    # Retrieving quadpy elemnts
    elements = _get_quadpy_elements(g, g_rot)
    
    # Retrieve pressure coefficients
    p_coeffs = d["error_estimates"]["ph"].copy()
    
    # Declaring integration methods
    if g.dim == 1:
        method = qp.line_segment.newton_cotes_closed(4)
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
        x = X#[0]
        
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
    ph_coeff = d_h["error_estimates"]["ph"].copy()
    ph_coeff = ph_coeff[cells_h]
    
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
    pl_coeff = d_l["error_estimates"]["ph"].copy()

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

#%% Testing the focking 

method = qp.line_segment.newton_cotes_closed(5)
num_points = method.points.size
normalized_intpts = (method.points+1)/2
weights = method.weights/2

# Faces of the higher-dimensional subdomain and cells of the lower-dimensional 
# subdomain, to be projected to the mortar grid
face_high, _, _,  = sps.find(g_m.mortar_to_master_avg())
face_high_centers = g_h.face_centers[:, face_high].T

# Find, to which cells "faces_high" belong to
cell_faces_map, cell_idx , _   = sps.find(g_h.cell_faces)
_, facehit, _ = np.intersect1d(cell_faces_map, face_high, return_indices=True)
cell_high = cell_idx[facehit]

face_nodes_map, _, _ = sps.find(g_h.face_nodes)
node_faces = face_nodes_map.reshape((np.array([g_h.num_faces, g_h.dim])))
node_faces_high = node_faces[face_high]
nodescoor_faces_high = np.zeros(
    [g_h.dim, node_faces_high.shape[0], node_faces_high.shape[1]]
    )
for dim in range(g_h.dim):
    nodescoor_faces_high[dim] = g_h.nodes[dim][node_faces_high]



# Reformat node coordinates to match size of integration point array
nodecoor_formatted  = np.empty([g_h.dim, face_high.size, num_points * g_h.dim])
for dim in range(g_h.dim):
    nodecoor_formatted[dim] = matlib.repeat(nodescoor_faces_high[dim], num_points, axis=1)

# Obtain evaluation points of the higher-dimensional faces 
faces_high_intcoor = np.empty([g_h.dim, face_high.size, num_points])
for dim in range(g_h.dim):
    faces_high_intcoor[dim] = (
        (nodecoor_formatted[dim][:, num_points:] - nodecoor_formatted[dim][:, :num_points])
        * normalized_intpts + nodecoor_formatted[dim][:, :num_points]
        )

# Retrieve postprocessed pressure coefficients
ph = d_h["error_estimates"]["ph"].copy()
ph_cell_high = ph[cell_high]

# Evaluate postprocessed higher dimensional pressure at integration points
tracep_intpts = _quadpyfy(ph_cell_high[:, 0], num_points)
for dim in range(g_h.dim):
    tracep_intpts += (
        _quadpyfy(ph_cell_high[:, dim+1], num_points) * faces_high_intcoor[dim]
        + _quadpyfy(ph_cell_high[:, -1], num_points)  * faces_high_intcoor[dim] ** 2
    )
    
#%% For the low-dimensional side, we have

# First, rotate the grid (whenever coordinates are used, the rotated
# version of the grids must be used).
gl_rot = rotate_embedded_grid(g_l)
cell_low,  _, _,  = sps.find(g_m.mortar_to_slave_avg())

cell_nodes_map, _, _ = sps.find(g_l.cell_nodes())
nodes_cell = cell_nodes_map.reshape(np.array([g_l.num_cells, g_l.dim + 1]))
nodes_cell_low = nodes_cell[cell_low]
nodescoor_cell_low = np.zeros(
    [g_l.dim, nodes_cell_low.shape[0], nodes_cell_low.shape[1]]
    )
for dim in range(g_l.dim):
    nodescoor_cell_low[dim] = gl_rot.nodes[dim][nodes_cell_low]

# Reformat node coordinates to match size of integration point array
nodecoor_format_low  = np.empty([g_l.dim, cell_low.size, num_points * (g_l.dim+1)])
for dim in range(g_l.dim):
    nodecoor_format_low[dim] = matlib.repeat(nodescoor_cell_low[dim], num_points, axis=1)

# Obtain evaluation from the higher-dimensional faces 
cells_low_intcoor = np.empty([g_l.dim, cell_low.size, num_points])
for dim in range(g_l.dim):
    cells_low_intcoor[dim] = (
        (nodecoor_format_low[dim][:, num_points:] - nodecoor_format_low[dim][:, :num_points])
        * normalized_intpts + nodecoor_format_low[dim][:, :num_points]
        )

# Rotate reconstructed pressure 
pl = d_l["error_estimates"]["ph"].copy()
pl_cell_low = pl[cell_low]

# Obtain reconstructed cell-center pressures
plow_intpts = _quadpyfy(pl_cell_low[:, 0], num_points)
for dim in range(g_l.dim):
    plow_intpts += (
        _quadpyfy(pl_cell_low[:, dim+1], num_points) * cells_low_intcoor[dim]
        + _quadpyfy(pl_cell_low[:, -1], num_points) * cells_low_intcoor[dim] ** 2
    )

#%% Obtaining the mortar solution
lam = d_e[pp.STATE][edge_variable]
# Compute the mortar velocities
mortar_highdim_areas =  g_m.master_to_mortar_avg() * g_h.face_areas
lam_vel = lam / mortar_highdim_areas
lam_vel_format = _quadpyfy(lam_vel, num_points)
V = g_m.cell_volumes
weights_format = matlib.repmat(weights, lam_vel.size, 1)
integrand = (lam_vel_format + (plow_intpts - tracep_intpts)) ** 2
integral = V * (integrand * weights_format).sum(axis=1)


#%% 
#from error_estimates_reconstruction import _oswald_with_bubbles
#_oswald_with_bubbles(g_1d, g_rot, d_1d, parameter_keyword)



#%% Cartesian to barycentric
# g = g_2d.copy()
# g_rot = rotate_embedded_grid(g)

# # Retrieving topological data
# cell_nodes_map, _, _ = sps.find(g.cell_nodes())
# nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
# nodes_coor_cell = np.empty([g.dim, nodes_cell.shape[0], nodes_cell.shape[1]])
# for dim in range(g.dim):
#     nodes_coor_cell[dim] = g_rot.nodes[dim][nodes_cell]

# # Assembling the local matrix for each cell
# lcl = nodes_coor_cell[0].flatten()
# for dim in range(g.dim-1):
#     lcl = np.vstack([lcl, nodes_coor_cell[dim+1].flatten()])
# lcl = np.vstack([lcl, np.ones(g.num_cells * (g.dim + 1))]).T
# lcl = np.reshape(lcl, [g.num_cells, g.dim + 1, g.dim + 1])

# # Obtaining the matrix M
# inv_mat = np.empty([g.num_cells, (g.dim+1) ** 2])
# for cell in range(g.num_cells):
#     inv_mat[cell] = np.linalg.inv(lcl[cell].T).flatten()
    
# def integrand_bubble_1d(X):
    
#     # Local coordinates
#     x = X[0]
    
#     # Constat from the bubble function with local compact support
#     constant = 4
    
#     # Elements of the matrix M
#     r11 = _quadpyfy(inv_mat[:, 0], int_point)
#     r12 = _quadpyfy(inv_mat[:, 1], int_point)
#     r21 = _quadpyfy(inv_mat[:, 2], int_point)
#     r22 = _quadpyfy(inv_mat[:, 3], int_point)

#     # Barycentric coordinates
#     b1  = r11 * x + r12
#     b2  = r21 * x + r22
    
#     # The integral is the constant times the product of the 
#     # barycentric coordinates
#     int_ = constant * (b1 * b2)
    
#     return int_

# def integrand_bubble_2d(X):
    
#     # Local coordinates
#     x = X[0]
#     y = X[1]
    
#     # Constat from the bubble function with local compact support
#     constant = 27
    
#     # Elements of the matrix M
#     r11 = _quadpyfy(inv_mat[:, 0], int_point)
#     r12 = _quadpyfy(inv_mat[:, 1], int_point)
#     r13 = _quadpyfy(inv_mat[:, 2], int_point)
#     r21 = _quadpyfy(inv_mat[:, 3], int_point)
#     r22 = _quadpyfy(inv_mat[:, 4], int_point)
#     r23 = _quadpyfy(inv_mat[:, 5], int_point)
#     r31 = _quadpyfy(inv_mat[:, 6], int_point)
#     r32 = _quadpyfy(inv_mat[:, 7], int_point)
#     r33 = _quadpyfy(inv_mat[:, 8], int_point)

#     # Barycentric coordinates
#     b1  = r11 * x + r12 * y + r13
#     b2  = r21 * x + r22 * y + r23
#     b3  = r31 * x + r32 * y + r33
    
#     # The integral is the constant times the product of the 
#     # barycentric coordinates
#     int_ = constant * (b1 * b2 * b3)
    
#     return int_

# def integrand_bubble_3d(X):
#     # Local coordinates
#     x = X[0]
#     y = X[1]
#     z = X[2]
    
#     # Constat from the bubble function with local compact support
#     constant = 256
    
#     # Elements of the matrix M
#     r11 = _quadpyfy(inv_mat[:, 0], int_point)
#     r12 = _quadpyfy(inv_mat[:, 1], int_point)
#     r13 = _quadpyfy(inv_mat[:, 2], int_point)
#     r14 = _quadpyfy(inv_mat[:, 3], int_point)
#     r21 = _quadpyfy(inv_mat[:, 4], int_point)
#     r22 = _quadpyfy(inv_mat[:, 5], int_point)
#     r23 = _quadpyfy(inv_mat[:, 6], int_point)
#     r24 = _quadpyfy(inv_mat[:, 7], int_point)
#     r31 = _quadpyfy(inv_mat[:, 8], int_point)
#     r32 = _quadpyfy(inv_mat[:, 9], int_point)
#     r33 = _quadpyfy(inv_mat[:, 10], int_point)
#     r34 = _quadpyfy(inv_mat[:, 11], int_point)
#     r41 = _quadpyfy(inv_mat[:, 12], int_point)
#     r42 = _quadpyfy(inv_mat[:, 13], int_point)
#     r43 = _quadpyfy(inv_mat[:, 14], int_point)
#     r44 = _quadpyfy(inv_mat[:, 15], int_point)

#     # Barycentric coordinates
#     b1  = r11 * x + r12 * y + r13 * z + r14
#     b2  = r21 * x + r22 * y + r23 * z + r24
#     b3  = r31 * x + r32 * y + r33 * z + r34
#     b4  = r41 * x + r42 * y + r43 * z + r44
    
#     # The integral is the constant times the product of the 
#     # barycentric coordinates
#     int_ = constant * (b1 * b2 * b3 * b4)
    
#     return int_

#%% 

# x, y, z, k = sym.symbols('x, y, z, k')

# (r11, r12, r13, r14, 
#  r21, r22, r23, r24,
#  r31, r32, r33, r34,
#  r41, r42, r43, r44) = sym.symbols('r11, r12, r13, r14, \
#                                     r21, r22, r23, r24, \
#                                     r31, r32, r33, r34, \
#                                     r41, r42, r43, r44')

# b1 = r11 * x + r12 * y + r13 * z + r14
# b2 = r21 * x + r22 * y + r23 * z + r24
# b3 = r31 * x + r32 * y + r33 * z + r34
# b4 = r41 * x + r42 * y + r43 * z + r44

# bK = 256 * k * b1 * b2 * b3 * b4

# bK_ex = bK.expand()

# # For g.dim == 1
# # The polynomial is given by:
# #   c0 * x**2 + c1 * x + c2
# c0 = (4*k*r11*r21).sum()
# c1 = (4*k*r11*r22 + 4*k*r12*r21).sum()
# c2 = (4*k*r12*r22).sum()

# # For g.dim == 2
# # The polynomial is given by:
# #   c0 * x**3 + c1 * x**2 * y + c2 * x**2 + c3 * x * y**2 + c4 * x * y
# #    + c5 * x + c6 * y**3 + c7 * y**2 + c8 * y + c9 
# c0 = (27*k*r11*r21*r31).sum()
# c1 = (27*k*r11*r21*r32 + 27*k*r11*r22*r31 + 27*k*r12*r21*r31).sum()
# c2 = (27*k*r11*r21*r33 + 27*k*r11*r23*r31 + 27*k*r13*r21*r31).sum()
# c3 = (27*k*r11*r22*r32 + 27*k*r12*r21*r32 + 27*k*r12*r22*r31).sum()
# c4 = (27*k*r11*r22*r33 + 27*k*r11*r23*r32 + 27*k*r12*r21*r33 
#      + 27*k*r12*r23*r31 + 27*k*r13*r21*r32 + 27*k*r13*r22*r31).sum()
# c5 = (27*k*r11*r23*r33 + 27*k*r13*r21*r33 + 27*k*r13*r23*r31).sum()
# c6 = (27*k*r12*r22*r32).sum()
# c7 = (27*k*r12*r22*r33 + 27*k*r12*r23*r32 + 27*k*r13*r22*r32).sum()
# c8 = (27*k*r12*r23*r33 + 27*k*r13*r22*r33 + 27*k*r13*r23*r32).sum()
# c9 = (27*k*r13*r23*r33).sum()

# # For g.dim == 3
# # The polynomial is given by:
# #
# c0

#%% OSWALD INTERPOLATOR FOR 2D

g = g_2d.copy()
g_rot = rotate_embedded_grid(g)
d = d_2d.copy()
kw = parameter_keyword
sd_operator_name = subdomain_operator_keyword
p_name = subdomain_variable

import error_estimates_utility as utils

ph = utils.get_postp_coeff(g, g_rot, d, kw, sd_operator_name, p_name)
d["error_estimates"]["ph"] = ph.copy()

# Retrieve coefficients of the postprocessed pressure
if "ph" in d["error_estimates"]:
    ph = d["error_estimates"]["ph"].copy()
else:
    raise ValueError("Pressure solution must be postprocessed first.")
      
# Mappings
cell_nodes_map, _, _ = sps.find(g.cell_nodes())
cell_faces_map, _, _ = sps.find(g.cell_faces)
nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

# ---------------------  Treatment of the nodes  -----------------------------

# Evaluate post-processed pressure at the nodes
nodes_p = np.zeros([g.num_cells, 3])
nx = g_rot.nodes[0][nodes_cell]  # local node x-coordinates
ny = g_rot.nodes[1][nodes_cell]  # local node y-coordinates

# Compute node pressures
for col in range(g.dim+1):      

    nodes_p[:, col] = (
          ph[:, 0] * nx[:, col] ** 2          # c0 * x ** 2
        + ph[:, 1] * nx[:, col] * ny[:, col]  # c1 * x * y
        + ph[:, 2] * nx[:, col]               # c2 * x
        + ph[:, 3] * ny[:, col] ** 2          # c3 * y ** 2
        + ph[:, 4] * ny[:, col]               # c4 * x
        + ph[:, 5]                            # c5 * 1
    )

# Average nodal pressure
node_cardinality = np.bincount(cell_nodes_map)
node_pressure = np.zeros(g.num_nodes)
for col in range(g.dim + 1):
    node_pressure += np.bincount(
        nodes_cell[:, col], 
        weights=nodes_p[:, col], 
        minlength=g.num_nodes
    )
node_pressure /= node_cardinality

# ---------------------  Treatment of the faces  -----------------------------

# Evaluate post-processed pressure at the face-centers
faces_p = np.zeros([g.num_cells, 3])
fx = g_rot.face_centers[0][faces_cell] # local face-center x-coordinates
fy = g_rot.face_centers[1][faces_cell] # local face-center y-coordinates

for col in range(g.dim+1):
    
    faces_p[:, col] = (
          ph[:, 0] * fx[:, col] ** 2          # c0 * x ** 2
        + ph[:, 1] * fx[:, col] * fy[:, col]  # c1 * x * y
        + ph[:, 2] * fx[:, col]               # c2 * x
        + ph[:, 3] * fy[:, col] ** 2          # c3 * y ** 2
        + ph[:, 4] * fy[:, col]               # c4 * x
        + ph[:, 5]                            # c5 * 1
    )

# Average face pressure
face_cardinality = np.bincount(cell_faces_map)
face_pressure = np.zeros(g.num_faces)
for col in range(3):
    face_pressure += np.bincount(
        faces_cell[:, col],
        weights=faces_p[:, col],
        minlength=g.num_faces
        )
face_pressure /= face_cardinality

# ---------------------- Treatment of the boundary points --------------------

bc = d[pp.PARAMETERS][kw]["bc"]
bc_values = d[pp.PARAMETERS][kw]["bc_values"]
# TODO: CHECK WHAT'S THE SITUATION WHEN bulk is 3D
# If external boundary face is Dirichlet, we overwrite the value,
# If external boundary face is Neumann, we leave it as it is.
external_dir_bound_faces = np.logical_and(bc.is_dir, g.tags["domain_boundary_faces"])
external_dir_bound_faces_vals = bc_values[external_dir_bound_faces]
face_pressure[external_dir_bound_faces] = external_dir_bound_faces_vals

# Now the nodes
face_vec = np.zeros(g.num_faces)
face_vec[external_dir_bound_faces] = 1
num_dir_face_of_node = g.face_nodes * face_vec
is_dir_node = num_dir_face_of_node > 0
face_vec *= 0
face_vec[external_dir_bound_faces] = bc_values[external_dir_bound_faces]
node_val_dir = g.face_nodes * face_vec
node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
node_pressure[is_dir_node] = node_val_dir[is_dir_node]

# ---------------------- Prepare for exporting -------------------------------

point_pressures      = np.column_stack([node_pressure[nodes_cell], face_pressure[faces_cell]])
point_coordinates    = np.empty([g.dim, g.num_cells, 6])
point_coordinates[0] = np.column_stack([nx, fx]) 
point_coordinates[1] = np.column_stack([ny, fy]) 



















