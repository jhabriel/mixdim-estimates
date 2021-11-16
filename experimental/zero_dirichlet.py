import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import mdestimates as mde

import mdestimates.estimates_utils as utils

def exact_source(grid):
    int_method = qp.t2.get_good_scheme(4)
    elements = utils.get_quadpy_elements(grid, grid)
    def integrand(x):
        return f_ex_np(x[0], x[1])
    return int_method.integrate(integrand, elements)

def div_u_mean_val(grid, data, estimate):
    # Retrieve reconstructed velocity
    recon_u = data[estimate.estimates_kw]["recon_u"].copy()
    u = utils.poly2col(recon_u)
    div_u = 2 * u[0]
    return div_u.flatten()

def f_mean_val(grid, data, estimate):
    volumes = grid.cell_volumes
    int_method = qp.t2.get_good_scheme(4)
    elements = utils.get_quadpy_elements(grid, grid)
    def integrand(x):
        return f_ex_np(x[0], x[1])
    integral = (1/volumes) * int_method.integrate(integrand, elements)
    return integral

def compute_residual_error(grid, data, estimate):

    # Retrieve reconstructed velocity
    recon_u = data[estimate.estimates_kw]["recon_u"].copy()
    u = utils.poly2col(recon_u)
    div_u = 2 * u[0]
    int_method = qp.t2.get_good_scheme(4)  # since f is quadratic, we need at least order 4
    elements = utils.get_quadpy_elements(grid, grid)

    # We now declare the different integrand regions and compute the norms
    def integrand(x):
        return (f_ex_np(x[0], x[1]) - div_u) ** 2
    integral = int_method.integrate(integrand, elements)

    return integral


#%% Analytical solution
x, y = sym.symbols("x y")
p_ex_sym = sym.sin(x*sym.pi) * sym.cos(y*sym.pi)
u_ex_sym = [-sym.diff(p_ex_sym, x), -sym.diff(p_ex_sym, y)]
f_ex_sym = sym.diff(u_ex_sym[0], x) + sym.diff(u_ex_sym[1], y)

p_ex_np = sym.lambdify((x, y), p_ex_sym, "numpy")
u_ex_np = [
    sym.lambdify((x, y), u_ex_sym[0], "numpy"),
    sym.lambdify((x, y), u_ex_sym[1], "numpy")
]
f_ex_np = sym.lambdify((x, y), f_ex_sym, "numpy")

#%% Create grid
# Set domain boundaries
domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
# Create network
network_2d = pp.FractureNetwork2d(None, None, domain)
# Set preferred mesh size
h = 1/10
mesh_args = {'mesh_size_frac': h, 'mesh_size_bound': h}
# Generate a mixed-dimensional mesh
gb = network_2d.mesh(mesh_args)
#n = 20
#gb = pp.meshing.cart_grid([], nx=[n, n], physdims=[1, 1])
# Extract grid and data dictionaries
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)
pp.set_state(d)

#%% Set up the problem
parameter_keyword = "flow"

# SPECIFY DATA
top = np.where(np.abs(g.face_centers[1] - 1) < 1e-5)[0]
bottom = np.where(np.abs(g.face_centers[1]) < 1e-5)[0]
right = np.where(np.abs(g.face_centers[0] - 1) < 1e-5)[0]
left = np.where(np.abs(g.face_centers[0]) < 1e-5)[0]
# On the left and right boundaries, we set homogeneous Neumann conditions
# Neumann conditions are set by default, so there is no need to do anything
# Define BoundaryCondition object
bc_faces = np.hstack((left, bottom, top, right))
bc_type = bc_faces.size * ['dir']
bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
# Alse set the values - specified as vector of size g.num_faces
bc_values = np.zeros(g.num_faces)
# (Integrated) source terms are given by the exact solution
source = f_ex_np(g.cell_centers[0], g.cell_centers[1]) * g.cell_volumes
specified_parameters = {"bc": bc, "bc_values": bc_values, "source": source}
pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

# DISCRETIZE
subdomain_discretization = pp.Tpfa(keyword=parameter_keyword)
source_discretization = pp.ScalarSource(keyword=parameter_keyword)
subdomain_variable = "pressure"
flux_variable = "flux"
subdomain_operator_keyword = "diffusion"
d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
d[pp.DISCRETIZATION] = {
    subdomain_variable: {
        subdomain_operator_keyword: subdomain_discretization,
        "source": source_discretization,
    }
}

# ASSEMBLE AND SOLVE
assembler = pp.Assembler(gb)
assembler.discretize()
A, b = assembler.assemble_matrix_rhs()
sol = sps.linalg.spsolve(A, b)
assembler.distribute_variable(sol)

# GET PRESSURE AND FLUXES
pp.fvutils.compute_darcy_flux(gb)
flux = d[pp.PARAMETERS]["flow"]["darcy_flux"]


#%% Solve eigenvalue problem
M = sps.eye(m=A.shape[0], n=A.shape[1])
#A1 = n ** 2 * A
A1 = g.num_cells * A
eig_min = np.real(sps.linalg.eigs(A1, k=1, which="SM")[0])[0]
C_p = 1/np.sqrt(eig_min)
print(f"Minimum eigvenvalue is: {eig_min}")
print(f"Poincare constant is: {C_p}")

#%% Obtain error estimates
# NOTE: Residual errors must be obtained separately!
estimates = mde.ErrorEstimate(gb)
estimates.estimate_error()
estimates.transfer_error_to_state()
kwe = estimates.estimates_kw
diffusive_squared = d[kwe]["diffusive_error"]
diffusive_error = np.sqrt(diffusive_squared.sum())
residual_error_squared = compute_residual_error(g, d, estimates)
residual_error_NC = 1/(np.sqrt(2)*np.pi) * np.sqrt(residual_error_squared.sum())
residual_error_LC = np.sum((g.cell_diameters()/np.pi)**2 * residual_error_squared) ** 0.5
majorant_NC = diffusive_error + residual_error_NC
majorant_LC = diffusive_error + residual_error_LC
print(f"Majorant LC: {majorant_LC}")
print(f"Majorant NC: {majorant_NC}")
print(f"Diffusive error: {diffusive_error}")
print(f"Residual error LC: {residual_error_LC}")
print(f"Residual error NC: {residual_error_NC}")

mean_divu = div_u_mean_val(g, d, estimates)
mean_f = f_mean_val(g, d, estimates)