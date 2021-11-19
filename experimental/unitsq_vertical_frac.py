import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import mdestimates as mde
import matplotlib.pyplot as plt
import os

import mdestimates.estimates_utils as utils
import model_utils as mutils
from mdestimates._velocity_reconstruction import _internal_source_term_contribution as mortar_jump
from analytical_2d import ExactSolution2D
from true_errors_2d import TrueErrors2D

#%% Create mesh
h = 1/160
gb = mutils.make_constrained_mesh(h=h)
g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)[0]
d_2d = gb.node_props(g_2d)
d_1d = gb.node_props(g_1d)
d_e = gb.edge_props((g_2d, g_1d))
mg = d_e["mortar_grid"]

#%% Set states
for _, d in gb:
    pp.set_state(d)

for _, d in gb.edges():
    pp.set_state(d)

#%% Retrieve boundary and cell indices
# cell_idx_list, regions_2d = mutils.get_2d_cell_indices(g_2d)
# bound_idx_list = mutils.get_2d_boundary_indices(g_2d)
#
# #%% Retrieve analytical expressions
# p2d_sym_list, p2d_numpy_list, p2d_cc = mutils.get_exact_2d_pressure(g_2d)
# gradp2d_sym_list, gradp2d_numpy_list, gradp2d_cc = mutils.get_exact_2d_pressure_gradient(
#     g_2d, p2d_sym_list
# )
# u2d_sym_list, u2d_numpy_list, u2d_cc = mutils.get_exact_2d_velocity(g_2d, gradp2d_sym_list)
# f2d_sym_list, f2d_numpy_list, f2d_cc = mutils.get_exact_2d_source_term(g_2d, u2d_sym_list)
# bc_vals_2d = mutils.get_2d_boundary_values(g_2d, bound_idx_list, p2d_numpy_list)

#%% Compute exact source terms
#integrated_f2d = mutils.integrate_source_2d(g_2d, f2d_numpy_list, cell_idx_list)
#integrated_f1d = integrate_source_1d(g_1d)
ex = ExactSolution2D(gb)

#%% Set parameters
parameter_keyword = "flow"

# Loop over the nodes
for g, d in gb:
    if g.dim == gb.dim_max():
        # All Dirichlet for the bulk
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        # Specificy bc values
        bc_values = ex.dir_bc_values()
        # Specifiy source terms
        source = ex.integrate_f2d()
        # Specified parameters dictionary
        specified_parameters = {"bc": bc, "bc_values": bc_values, "source": source}
    else:
        # Specifiy source terms
        source = ex.integrate_f1d()
        # Specified parameters dictionary
        specified_parameters = {"source": source}
    pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

# Loop over the edges
for e, d in gb.edges():
    data = {"normal_diffusivity": 1.0}
    mg = d["mortar_grid"]
    pp.initialize_data(mg, d, parameter_keyword, data)

#%% Discretization
subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)
source_discretization = pp.ScalarSource(keyword=parameter_keyword)
subdomain_variable = "pressure"
flux_variable = "flux"
subdomain_operator_keyword = "diffusion"
edge_discretization = pp.RobinCoupling(
    parameter_keyword, subdomain_discretization, subdomain_discretization
)
edge_variable = "interface_flux"
coupling_operator_keyword = "interface_diffusion"

# Loop over the nodes
for g, d in gb:
    d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
    d[pp.DISCRETIZATION] = {
        subdomain_variable: {
            subdomain_operator_keyword: subdomain_discretization,
            "source": source_discretization,
        }
    }

# Loop over the edges
for e, d in gb.edges():
    # Get the grids of the neighboring subdomains
    g1, g2 = gb.nodes_of_edge(e)
    # The interface variable has one degree of freedom per cell in the mortar grid
    d[pp.PRIMARY_VARIABLES] = {edge_variable: {"cells": 1}}
    # The coupling discretization links an edge discretization with variables
    d[pp.COUPLING_DISCRETIZATION] = {
        coupling_operator_keyword: {
            g1: (subdomain_variable, subdomain_operator_keyword),
            g2: (subdomain_variable, subdomain_operator_keyword),
            e: (edge_variable, edge_discretization),
        }
    }

#%% Obtain numerical solution
assembler = pp.Assembler(gb)
assembler.discretize()
A, b = assembler.assemble_matrix_rhs()
sol = sps.linalg.spsolve(A, b)
assembler.distribute_variable(sol)

#%% Obtain error estimates
estimates = mde.ErrorEstimate(gb, lam_name=edge_variable)
estimates.estimate_error()
estimates.transfer_error_to_state()
kwe = estimates.estimates_kw
#estimates.print_summary(scaled=False)

#%% Compute error estimates
# Get hold of diffusive errors
desq_2d = d_2d[pp.STATE]["diffusive_error"]
desq_1d = d_1d[pp.STATE]["diffusive_error"]
desq_mortar = d_e[pp.STATE]["diffusive_error"]
diffusive_error_2d = desq_2d.sum() ** 0.5
diffusive_error_1d = desq_1d.sum() ** 0.5
diffusive_error_mortar = desq_mortar.sum() ** 0.5
diffusive_error = diffusive_error_2d + diffusive_error_1d + diffusive_error_mortar
print(f"Diffusive error 2D: {diffusive_error_2d}")
print(f"Diffusive error 1D: {diffusive_error_1d}")
print(f"Diffusive error mortar: {diffusive_error_mortar}")
print(f"Diffusive error: {diffusive_error}")
print(50*"-")

# Get hold of residual errors
te = TrueErrors2D(gb, estimates)
resq_2d = te.residual_error_squared_2d()
resq_1d = te.residual_error_squared_1d()
residual_error_2d = resq_2d.sum() ** 0.5
residual_error_1d = resq_1d.sum() ** 0.5
residual_error_NC = te.residual_error_global_poincare()
residual_error_LC = te.residual_error_local_poincare()
print(f"Residual error 2d: {residual_error_2d}")
print(f"Residual error 1d: {residual_error_1d}")
print(f"Residual error NC: {residual_error_NC}")
print(f"Residual error LC: {residual_error_LC}")
print(50*"-")

# Obtain majorant
majorant_NC = diffusive_error + residual_error_NC
majorant_LC = diffusive_error + residual_error_LC
print(f"Majorant NC: {majorant_NC}")
print(f"Majorant LC: {majorant_LC}")

# Obtain true pressure error
true_pressure_error = te.pressure_error()
true_velocity_error = te.velocity_error()
print(f"True pressure error: {true_pressure_error}")
print(f"True velocitiy error: {true_velocity_error}")

# Obtain efficiency indices; pressure and velocity
efficiency_pressure_NC = majorant_NC / true_pressure_error
efficiency_pressure_LC = majorant_LC / true_pressure_error
efficiency_velocity_NC = majorant_NC / true_velocity_error
efficiency_velocity_LC = majorant_LC / true_velocity_error
efficiency_combined_NC = 3*majorant_NC / (true_pressure_error + true_velocity_error +
                                          residual_error_NC)
efficiency_combined_LC = 3*majorant_LC / (true_pressure_error + true_velocity_error +
                                          residual_error_LC)
print(f"Efficiency index NC (pressure): {efficiency_pressure_NC}")
print(f"Efficiency index LC (pressure): {efficiency_pressure_LC}")
print(f"Efficiency index NC (velocity): {efficiency_velocity_NC}")
print(f"Efficiency index LC (velocity): {efficiency_velocity_LC}")
print(f"Efficiency index NC (combined): {efficiency_combined_NC}")
print(f"Efficiency index LC (combined): {efficiency_combined_LC}")
print(50*"-")

#%% Print true pressure errors
true_pressure_error_2d = te.pressure_error_squared_2d().sum() ** 0.5
true_pressure_error_1d = te.pressure_error_squared_1d().sum() ** 0.5
true_pressure_error_mortar = te.pressure_error_squared_mortar().sum() ** 0.5
print(f"True pressure error 2D: {true_pressure_error_2d}")
print(f"True pressure error 1D: {true_pressure_error_1d}")
print(f"True pressure error mortar: {true_pressure_error_mortar}")
true_velocity_error_2d = te.velocity_error_squared_2d().sum() ** 0.5
true_velocity_error_1d = te.velocity_error_squared_1d().sum() ** 0.5
true_velocity_error_mortar = te.velocity_error_squared_mortar().sum() ** 0.5
print(f"True velocity error 2D: {true_velocity_error_2d}")
print(f"True velocity error 1D: {true_velocity_error_1d}")
print(f"True velocity error mortar: {true_velocity_error_mortar}")

# #%% Pressure
# g_rot = utils.rotate_embedded_grid(g_1d)
# p1d_cc = te.p1d()
# coeffs = d_1d["estimates"]["recon_p"].copy()
# reconp_cc = coeffs[:, 0] * g_rot.cell_centers[0] + coeffs[:, 1]
# plt.plot(p1d_cc, label="Exact Pressure")
# plt.plot(reconp_cc, label="Reconstructed Pressure")
# plt.legend()
# plt.show()
#
# #%% Pressure gradient
# gradp1d_cc = np.array([np.zeros(g_1d.num_cells), te.gradp1d(), np.zeros(g_1d.num_cells)])
# gradp1d_cc_rotated = np.dot(g_rot.rotation_matrix, gradp1d_cc)[g_rot.dim_bool]
# recongradp_cc = coeffs[:, 0]
# plt.plot(gradp1d_cc_rotated[0], label="Exact Pressure Gradient")
# plt.plot(recongradp_cc, label="Reconstructed Pressure Gradient")
# plt.legend()
# plt.show()
#
# #%% Velocity reconstruction
# # Reconstructed cell center velocities
# coeffs = d_1d["estimates"]["recon_u"].copy()
#
# u_recon_cc = coeffs[:, 0] * g_rot.cell_centers[0] + coeffs[:, 1]
# # Rotated exact velocities
# u1_cc = np.array([np.zeros(g_1d.num_cells), te.u1d(), np.zeros(g_1d.num_cells)])
# u_cc_rot = np.dot(g_rot.rotation_matrix, u1_cc)[g_rot.dim_bool]
#
#
# #%% PLOTS
# plt.title(f"Mesh size: {h}")
# plt.plot(te.p1d(), label="Exact p")
# plt.plot(te.reconstructed_p1d(), label="Recon p")
# plt.legend()
# plt.show()
#
# plt.title(f"Mesh size: {h}")
# plt.plot(te.gradp1d(), label="Exact gradp")
# plt.plot(te.reconstructed_gradp1()[0], label="Recon gradp")
# #plt.plot(-1 * te.reconstructed_u1d()[0], label="-recon u")
# plt.legend()
# plt.show()
#
# # pp.plot_grid(g_2d, ex.p2d(), plot_2d=True)
# # pp.plot_grid(g_2d, ex.f2d(), plot_2d=True)
# # pp.plot_grid(g_2d, residual_error_squared, plot_2d=True)
# # pp.plot_grid(g_2d, diffusive_error_squared, plot_2d=True)
#
# #%%
# plt.plot(te.lmbda()[:20], mg.cell_centers[1][:20])
# plt.xlabel("Mortar flux")
# plt.ylabel("y")
# plt.show()
#
# plt.plot(te.p1d(), g_1d.cell_centers[1])
# plt.xlabel("Fracture pressure")
# plt.ylabel("y")
# plt.show()
#
# plt.plot(te.u1d(), g_1d.cell_centers[1])
# plt.xlabel("Fracture flux")
# plt.ylabel("y")
# plt.show()
#
# plt.plot(te.f1d(), g_1d.cell_centers[1])
# plt.xlabel("Source term in the fracture")
# plt.ylabel("y")
# plt.show()

#%% Print into latex

# Pressure in the matrix
p2_bot = "p2_bot = " + sym.latex(te.p2d("sym")[0]) + "\n"
p2_mid = "p2_mid = " + sym.latex(te.p2d("sym")[1]) + "\n"
p2_top = "p2_top = " + sym.latex(te.p2d("sym")[2]) + "\n"
p2_txt = [p2_bot, p2_mid, p2_top]

# Velocity in the matrix
u2_bot_x = "u2_bot_x = " + sym.latex(te.u2d("sym")[0][0]) + "\n"
u2_bot_y = "u2_bot_y = " + sym.latex(te.u2d("sym")[0][1]) + "\n"
u2_mid_x = "u2_mid_x = " + sym.latex(te.u2d("sym")[1][0]) + "\n"
u2_mid_y = "u2_mid_y = " + sym.latex(te.u2d("sym")[1][1]) + "\n"
u2_top_x = "u2_top_x = " + sym.latex(te.u2d("sym")[2][0]) + "\n"
u2_top_y = "u2_top_y = " + sym.latex(te.u2d("sym")[2][1]) + "\n"
u2_txt = [u2_bot_x, u2_bot_y, u2_mid_x, u2_mid_y, u2_top_x, u2_top_y]

# Source in the matrix
f2_bot = "f2_bot = " + sym.latex(te.f2d("sym")[0])+ "\n"
f2_mid = "f2_mid = " + sym.latex(te.f2d("sym")[1]) + "\n"
f2_top = "f2_top = " + sym.latex(te.f2d("sym")[2]) + "\n"
f2_txt = [f2_bot, f2_mid, f2_top]

file_name = "sols.txt"
if os.path.exists(file_name):
    os.remove(file_name)

file = open("sols.txt", "a")

file.writelines(p2_txt)
file.write("\n")
file.writelines(u2_txt)
file.write("\n")
file.writelines(f2_txt)

file.close()

#%%

# Atomic symbols
x, y, z = sym.symbols("x y z", positive=True)
alpha = sym.symbols("alpha")
beta1, beta2 = sym.symbols("beta1, beta2")
gamma1, gamma2 = sym.symbols("gamma1, gamma2")
n = sym.symbols("n", positive=True)
dbot, dmid, dtop = sym.symbols("d_2^a d_2^b d_2^c", positive=True)
omega = sym.symbols("omega2", positive=True)

alpha_exp = x - sym.Rational(1, 2)
dalpha_dx = sym.Integer(1)
dalpha_dy = sym.Integer(0)
dalpha_dz = sym.Integer(0)

beta1_exp = y - sym.Rational(1, 4)
dbeta1_dx = sym.Integer(0)
dbeta1_dy = sym.Integer(1)
dbeta1_dz = sym.Integer(0)

beta2_exp = y - sym.Rational(3, 4)
dbeta2_dx = sym.Integer(0)
dbeta2_dy = sym.Integer(1)
dbeta2_dz = sym.Integer(0)

gamma1_exp = z - 0.25
dgamma1_dx = sym.Integer(0)
dgamma1_dy = sym.Integer(0)
dgamma1_dz = sym.Integer(1)

gamma2_exp = z - 0.75
dgamma2_dx = sym.Integer(0)
dgamma2_dy = sym.Integer(0)
dgamma2_dz = sym.Integer(1)

omega_exp = beta1 ** 2 + beta2 ** 2
domega_dbeta1 = sym.diff(omega_exp, beta1)
domega_dbeta2 = sym.diff(omega_exp, beta2)
domega_dx = domega_dbeta1 * dbeta1_dx + domega_dbeta2 * dbeta2_dx
domega_dy = domega_dbeta1 * dbeta1_dy + domega_dbeta2 * dbeta2_dy

dbot_exp = sym.sqrt(alpha ** 2 + beta1 ** 2)
ddbot_dalpha = sym.diff(dbot_exp, alpha)
ddbot_dbeta1 = sym.diff(dbot_exp, beta1)
ddbot_dx = ddbot_dalpha * dalpha_dx + ddbot_dbeta1 * dbeta1_dx
ddbot_dy = ddbot_dalpha * dalpha_dy + ddbot_dbeta1 * dbeta1_dy

dmid_exp = (alpha ** 2) ** 0.5
ddmid_dalpha = sym.diff(dmid_exp, alpha)
ddmid_dx = ddmid_dalpha * dalpha_dx
ddmid_dy = ddmid_dalpha * dalpha_dy

dtop_exp = (alpha ** 2 + beta2 ** 2) ** 0.5
ddtop_dalpha = sym.diff(dtop_exp, alpha)
ddtop_dbeta1 = sym.diff(dtop_exp, beta2)

# Derivatives of the pressure
pbot = dbot ** (n + 1)
dpbot_ddbot = sym.diff(pbot, dbot)
dpbot_dx = sym.simplify(dpbot_ddbot * ddbot_dx)
dpbot_dy = sym.simplify(dpbot_ddbot * ddbot_dy)

pmid = dmid ** (n + 1) + omega * dmid
dpmid_ddmid = sym.diff(pmid, dmid)
dpmid_domega = sym.diff(pmid, omega)
dpmid_dx = dpmid_ddmid * ddmid_dx + dpmid_domega * domega_dx
dpmid_dy = dpmid_ddmid * ddmid_dy + dpmid_domega * domega_dy

print([print(sym.latex(-dpbot_dx)), print(sym.latex(-dpbot_dy))])
print([print(sym.latex(-dpmid_dx)), print(sym.latex(-dpmid_dy))])



























