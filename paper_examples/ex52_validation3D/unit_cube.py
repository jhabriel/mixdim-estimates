import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import mdestimates as mde
import matplotlib.pyplot as plt
import os

from analytical_3d import ExactSolution3D
from true_errors_3d import TrueErrors3D

def make_constrained_mesh(mesh_size=0.2):
    """
    Creates an unstructured 3D mesh for a given target mesh size for the case
    of a  single 2D vertical fracture embedded in a 3D domain

    Parameters
    ----------
    mesh_size : float, optional
        Target mesh size. The default is 0.2.

    Returns
    -------
    gb : PorePy Object
        Porepy grid bucket object.

    """
    # Load fracture network: Fracture + Ghost Fractures
    network_3d = pp.fracture_importer.network_3d_from_csv("network.csv")

    # Create mesh_arg dictionary
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_bound": mesh_size,
        "mesh_size_min": mesh_size / 10,
    }

    # Construct grid bucket
    ghost_fracs = list(np.arange(1, 25))  # 1 to 24
    gb = network_3d.mesh(mesh_args, constraints=ghost_fracs)

    return gb


#%% Create mesh
h = 1/10
gb = make_constrained_mesh(mesh_size=h)
g_3d = gb.grids_of_dimension(3)[0]
g_2d = gb.grids_of_dimension(2)[0]
d_3d = gb.node_props(g_3d)
d_2d = gb.node_props(g_2d)
d_e = gb.edge_props((g_3d, g_2d))
mg = d_e["mortar_grid"]

#%% Set states
for _, d in gb:
    pp.set_state(d)

for _, d in gb.edges():
    pp.set_state(d)

ex = ExactSolution3D(gb)

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
        source = ex.integrate_f3d()
        # Specified parameters dictionary
        specified_parameters = {"bc": bc, "bc_values": bc_values, "source": source}
    else:
        # Specifiy source terms
        source = ex.integrate_f2d()
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
desq_3d = d_3d[pp.STATE]["diffusive_error"]
desq_2d = d_2d[pp.STATE]["diffusive_error"]
desq_mortar = d_e[pp.STATE]["diffusive_error"]
diffusive_error_3d = desq_3d.sum() ** 0.5
diffusive_error_2d = desq_2d.sum() ** 0.5
diffusive_error_mortar = desq_mortar.sum() ** 0.5
diffusive_error = diffusive_error_3d + diffusive_error_2d + \
                  diffusive_error_mortar
print(f"Diffusive error 2D: {diffusive_error_3d}")
print(f"Diffusive error 1D: {diffusive_error_2d}")
print(f"Diffusive error mortar: {diffusive_error_mortar}")
print(f"Diffusive error: {diffusive_error}")
print(50*"-")

# Get hold of residual errors
te = TrueErrors3D(gb, estimates)
residual_norm_3d = te.residual_norm_squared_3d().sum() ** 0.5
residual_norm_2d = te.residual_norm_squared_2d().sum() ** 0.5
residual_error_NC = te.residual_error_global_poincare()
residual_error_LC = te.residual_error_local_poincare()
print(f"Residual error 3d: {residual_norm_3d}")
print(f"Residual error 2d: {residual_norm_2d}")
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
true_pressure_error_3d = te.pressure_error_squared_3d().sum() ** 0.5
true_pressure_error_2d = te.pressure_error_squared_2d().sum() ** 0.5
true_pressure_error_mortar = te.pressure_error_squared_mortar().sum() ** 0.5
print(f"True pressure error 3D: {true_pressure_error_3d}")
print(f"True pressure error 2D: {true_pressure_error_2d}")
print(f"True pressure error mortar: {true_pressure_error_mortar}")
true_velocity_error_3d = te.velocity_error_squared_3d().sum() ** 0.5
true_velocity_error_2d = te.velocity_error_squared_2d().sum() ** 0.5
true_velocity_error_mortar = te.velocity_error_squared_mortar().sum() ** 0.5
print(f"True velocity error 2D: {true_velocity_error_3d}")
print(f"True velocity error 1D: {true_velocity_error_2d}")
print(f"True velocity error mortar: {true_velocity_error_mortar}")

# #%% Atomic symbols
# x, y = sym.symbols("x y")
# alpha = sym.symbols("alpha")
# beta1, beta2 = sym.symbols("beta1, beta2")
# gamma1, gamma2 = sym.symbols("gamma1, gamma2")
# n = sym.symbols("n")
# dbot, dmid, dtop = sym.symbols("d_2^a d_2^b d_2^c")
# omega = sym.symbols("omega2")
#
# alpha_exp = x - 0.5
# dalpha_dx = sym.Integer(1)
# dalpha_dy = sym.Integer(0)
#
# beta1_exp = y - 0.25
# dbeta1_dx = sym.Integer(0)
# dbeta1_dy = sym.Integer(1)
#
# beta2_exp = y - 0.75
# dbeta2_dx = sym.Integer(0)
# dbeta2_dy = sym.Integer(1)
#
# omega_exp = beta1 ** 2 * beta2 ** 2
# domega_dbeta1 = sym.diff(omega_exp, beta1)
# domega_dbeta2 = sym.diff(omega_exp, beta2)
# domega_dx = domega_dbeta1 * dbeta1_dx + domega_dbeta2 * dbeta2_dx
# domega_dy = domega_dbeta1 * dbeta1_dy + domega_dbeta2 * dbeta2_dy
#
# dbot_exp = (alpha ** 2 + beta1 ** 2) ** 0.5
# ddbot_dalpha = sym.diff(dbot_exp, alpha)
# ddbot_dbeta1 = sym.diff(dbot_exp, beta1)
# ddbot_dx = ddbot_dalpha * dalpha_dx + ddbot_dbeta1 * dbeta1_dx
# ddbot_dy = ddbot_dalpha * dalpha_dy + ddbot_dbeta1 * dbeta1_dy
#
# dmid_exp = (alpha ** 2) ** 0.5
# ddmid_dalpha = sym.diff(dmid_exp, alpha)
# ddmid_dx = ddmid_dalpha * dalpha_dx
# ddmid_dy = ddmid_dalpha * dalpha_dy
#
# dtop_exp = (alpha ** 2 + beta2 ** 2) ** 0.5
# ddtop_dalpha = sym.diff(dtop_exp, alpha)
# ddtop_dbeta2 = sym.diff(dtop_exp, beta2)
# ddtop_dx = ddtop_dalpha * dalpha_dx + ddtop_dbeta2 * dbeta2_dx
# ddtop_dy = ddtop_dalpha * dalpha_dy + ddtop_dbeta2 * dbeta2_dy
#
# # Derivatives of the pressure
# pbot = dbot ** (n + 1)
# dpbot_ddbot = sym.diff(pbot, dbot)
# dpbot_dx = sym.simplify(dpbot_ddbot * ddbot_dx)
# dpbot_dy = sym.simplify(dpbot_ddbot * ddbot_dy)
#
# pmid = dmid ** (n + 1) + omega * dmid
# dpmid_ddmid = sym.diff(pmid, dmid)
# dpmid_domega = sym.diff(pmid, omega)
# dpmid_dx = dpmid_ddmid * ddmid_dx + dpmid_domega * domega_dx
# dpmid_dy = dpmid_ddmid * ddmid_dy + dpmid_domega * domega_dy
#
# ptop = dtop ** (n + 1)
# dptop_ddtop = sym.diff(ptop, dtop)
# dptop_dx = sym.simplify(dptop_ddtop * ddtop_dx)
# dptop_dy = sym.simplify(dptop_ddtop * ddtop_dy)
#
# ubotx = - alpha * (n + 1) * dbot ** (n - 1)
# dubotx_dalpha = sym.diff(ubotx, alpha)
# dubotx_ddbot = sym.diff(ubotx, dbot)
# dubotx_dx = dubotx_dalpha * dalpha_dx + dubotx_ddbot * ddbot_dx
#
# uboty = - beta1 * (n + 1) * dbot ** (n - 1)
# duboty_dbeta1 = sym.diff(uboty, beta1)
# duboty_ddbot = sym.diff(uboty, dbot)
# duboty_dy = duboty_dbeta1 * dbeta1_dy + duboty_ddbot * ddbot_dy
#
# umidx = - (alpha ** 2) ** 0.5 * alpha ** (-1) * (omega + (n + 1) * dmid ** n)
# dumidx_dalpha = sym.diff(umidx, alpha)
# dumidx_domega = sym.diff(umidx, omega)
# dumidx_ddmid = sym.diff(umidx, dmid)
# dumidx_dx = dumidx_dalpha * dalpha_dx + dumidx_domega * domega_dx + dumidx_ddmid * ddmid_dx
#
# umidy = - dmid * (2 * beta1 ** 2 * beta2 + 2 * beta1 * beta2 ** 2)
# dumidy_ddmid = sym.diff(umidy, dmid)
# dumidy_dbeta1 = sym.diff(umidy, beta1)
# dumidy_dbeta2 = sym.diff(umidy, beta2)
# dumidy_dy = dumidy_ddmid * ddmid_dy + dumidy_dbeta1 * dbeta1_dy + dumidy_dbeta2 * dbeta2_dy
#
# utopx = -alpha * (n + 1) * dtop ** (n - 1)
# dutopx_dalpha = sym.diff(utopx, alpha)
# dutopx_ddtop = sym.diff(utopx, dtop)
# dutopx_dx = dutopx_dalpha * dalpha_dx + dutopx_ddtop * ddtop_dx
#
# utopy = - beta2 * (n + 1) * dtop ** (n - 1)
# dutopy_dbeta2 = sym.diff(utopy, beta2)
# dutopy_ddtop = sym.diff(utopy, dtop)
# dutopy_dy = dutopy_dbeta2 * dbeta2_dy + dutopy_ddtop * ddtop_dy
#
# fbot = sym.simplify(dubotx_dx + duboty_dy)
# fmid = sym.simplify(dumidx_dx + dumidy_dy)
# ftop = sym.simplify(dutopx_dx + dutopy_dy)
#
# #%%
# x, y = sym.symbols("x y")
# alpha = x - 0.5
# beta1 = y - 0.25
# beta2 = y - 0.75
# dbot = (alpha ** 2 + beta1 ** 2) ** 0.5
# dmid = (alpha ** 2) ** 0.5
# dtop = (alpha ** 2 + beta2 ** 2) ** 0.5
# omega = beta1 ** 2 * beta2**2
# n = 1.5
#
# # Test bulk pressure
# p2d = [dbot ** (n + 1), dmid ** (n + 1) + omega * dmid, dtop ** (n + 1)]
# for region in range(3):
#     val = p2d[region].subs([(x, 3.123), (y, -2.251)])
#     known = te.p2d("sym")[region].subs([(x, 3.123), (y, -2.251)])
#     np.testing.assert_almost_equal(val, known)
#
# # Test bulk velocity
# u2d = [
#     [
#         - alpha * (n + 1) * dbot ** (n - 1),
#         - beta1 * (n + 1) * dbot ** (n - 1)
#     ],
#     [
#         - (alpha ** 2) ** 0.5 * alpha ** (-1) * (omega + (n + 1) * dmid ** n),
#         - dmid * (2 * beta1 ** 2 * beta2 + 2 * beta1 * beta2 ** 2)
#     ],
#     [
#         - alpha * (n + 1) * dtop ** (n - 1),
#         - beta2 * (n + 1) * dtop ** (n - 1)
#     ],
#     ]
# for region in range(3):
#     for dim in range(g_2d.dim):
#         val = u2d[region][dim].subs([(x, 3.123), (y, -2.251)])
#         known = te.u2d("sym")[region][dim].subs([(x, 3.123), (y, -2.251)])
#         np.testing.assert_almost_equal(val, known)
#
# # Test bulk source term
# f2d = [
#     -(n + 1) * dbot ** (-2) * (alpha ** 2 * (n - 1) * dbot ** (n - 1)
#                                  + beta1 ** 2 * (n - 1) * dbot ** (n - 1)
#                                  + 2 * dbot ** (n + 1)
#                                  ),
#     (- 2 * dmid * (beta1 * (beta1 + 2 * beta2) + beta2 * (2 * beta1 + beta2))
#         - dmid ** (n - 1) * n * (n + 1)
#     ),
#     -(n + 1) * dtop ** (-2) * (alpha ** 2 * (n - 1) * dtop ** (n - 1)
#                                  + beta2 ** 2 * (n - 1) * dtop ** (n - 1)
#                                  + 2 * dtop ** (n + 1)
#                                  )
#     ]
# for region in range(3):
#     val = f2d[region].subs([(x, 3.123), (y, -2.251)])
#     known = te.f2d("sym")[region].subs([(x, 3.123), (y, -2.251)])
#     np.testing.assert_almost_equal(val, known)
#
# # Test mortar flux
# lmbda = omega
# val = lmbda.subs([(x, 3.123), (y, -2.251)])
# known = te.lmbda("sym").subs([(x, 3.123), (y, -2.251)])
# np.testing.assert_almost_equal(val, known)
#
# # Test fracture pressure
# p1d = - omega
# val = p1d.subs([(x, 3.123), (y, -2.251)])
# known = te.p1d("sym").subs([(x, 3.123), (y, -2.251)])
# np.testing.assert_almost_equal(val, known)
#
# # Test fracture velocity
# u1d = 2 * (beta1 ** 2 * beta2 + beta1 * beta2 ** 2)
# val = u1d.subs([(x, 3.123), (y, -2.251)])
# known = te.u1d("sym").subs([(x, 3.123), (y, -2.251)])
# np.testing.assert_almost_equal(val, known)
#
# # Test fracture source
# f1d = 8 * beta1 * beta2 + 2 * (beta1 ** 2 + beta2 ** 2) - 2 * omega
# val = f1d.subs([(x, 3.123), (y, -2.251)])
# known = te.f1d("sym").subs([(x, 3.123), (y, -2.251)])
# np.testing.assert_almost_equal(val, known)











