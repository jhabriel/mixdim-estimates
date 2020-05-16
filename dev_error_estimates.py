#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:18:39 2019

@author: eke001
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy

from examples.example1.benchmark1.test_vem import make_grid_bucket
from examples.example1.benchmark1.test_fv import add_data as add_flow_data

single_grid = False
dual_method = False

grid_variable = 'pressure'
mortar_variable = 'mortar_flux'

operator_keyword = 'diffusion'
coupling_operator_keyword = 'coupling_operator'

kw_f = 'flow'
kw_t = 'transport'

if dual_method:
    tpfa = pp.RT0(kw_f)   # Lowest order Raviart-Thomas
else:
    tpfa = pp.Mpfa(kw_f)  # Multi-Point Flux Approximation

if not single_grid:
    mesh_size = 0.1
    gb, domain = make_grid_bucket(mesh_size)
    fracture_permeability = 1e4

    add_flow_data(gb, domain, fracture_permeability, mesh_size)

    edge_discretization = pp.RobinCoupling(kw_f, tpfa, tpfa)

    # Loop over the nodes in the GridBucket, define primary variables and discretization schemes
    for g, d in gb:
        if dual_method:
            d[pp.PRIMARY_VARIABLES] = {grid_variable: {"cells": 1, "faces": 1}}
        else:
            d[pp.PRIMARY_VARIABLES] = {grid_variable: {"cells": 1, "faces": 0}}
        d[pp.DISCRETIZATION] = {grid_variable: {operator_keyword: tpfa}}

    # Loop over the edges in the GridBucket, define primary variables and discretizations
    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar_variable: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            coupling_operator_keyword: {
                g1: (grid_variable, operator_keyword),
                g2: (grid_variable, operator_keyword),
                e: (mortar_variable, edge_discretization),
            }
        }

    assembler = pp.Assembler(gb)

    # Assemble the linear system, using the information stored in the GridBucket
    A, b = assembler.assemble_matrix_rhs()
    block_dof = assembler.block_dof
    full_dof = assembler.full_dof

    pressure = sps.linalg.spsolve(A, b)

    #assembler.distribute_variable(gb, pressure, block_dof, full_dof)
    assembler.distribute_variable(pressure)

    if not dual_method:
        pp.fvutils.compute_darcy_flux(gb, lam_name=mortar_variable)
        for g, d in gb:
            if g.dim > 0:
                d['darcy_flux'] = d[pp.PARAMETERS][kw_f]['darcy_flux']
        for e, d in gb.edges():
            d['darcy_flux'] = d[pp.PARAMETERS][kw_f]['darcy_flux']
    else:
        for g, d in gb:
            if g.dim > 0:
                d['darcy_flux'] = tpfa.extract_flux(g, d['pressure'], None)
            d['pressure'] = tpfa.extract_pressure(g, d['pressure'], None)



    gb.assign_node_ordering()

    num_grids = gb.num_graph_nodes()

    node_pressure = np.empty(num_grids, dtype=np.object)
    face_pressure = np.empty_like(node_pressure)
    cell_pressure = np.empty_like(node_pressure)

    g = gb.grids_of_dimension(2)[0]
    data = gb.node_props(g)
    print('Minimum pressure in 2d domain ' + str(data[grid_variable].min()))
    if not dual_method:

        flux = data[pp.DISCRETIZATION_MATRICES][kw_f]['flux'] * data[grid_variable] + \
                data[pp.DISCRETIZATION_MATRICES][kw_f]['bound_flux'] * data[pp.PARAMETERS][kw_f]['bc_values']
        data[pp.PARAMETERS][kw_f]['full_flux'] = flux
    else:
        data[pp.PARAMETERS][kw_f]['darcy_flux'] = data['darcy_flux']
        data[pp.PARAMETERS][kw_f]['full_flux'] = data['darcy_flux']


if single_grid:

    neu_left = True

    nx = 40
    g = pp.StructuredTriangleGrid([nx, nx], [1, 1])
    g.compute_geometry()

    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    labels = np.array(["dir"] * bound_faces.size)

    xf = g.face_centers[:, bound_faces]
    if neu_left:
        hit = np.where(xf[0] < 1e-10)
        labels[hit] = "neu"


    bc = pp.BoundaryCondition(g, bound_faces, labels)

    x, y = sympy.symbols('x y')
    u = (1-x) * sympy.sin(x+1) * sympy.cos(np.pi * y)
    u_f = sympy.lambdify((x, y), u, 'numpy')
    dux = sympy.diff(u, x)
    duy = sympy.diff(u, y)
    dux_f = sympy.lambdify((x, y), dux, 'numpy')
    duy_f = sympy.lambdify((x, y), duy, 'numpy')
    rhs = -sympy.diff(dux, x) - sympy.diff(duy, y)
    rhs_f = sympy.lambdify((x, y), rhs, 'numpy')


    bc_val = np.zeros(g.num_faces)
    bc_val[bound_faces] = u_f(xf[0], xf[1])
    if neu_left:
        hit = np.where(xf[0] < 1e-10)
        bc_val[bound_faces[hit]] = dux_f(xf[0, hit], xf[1, hit]) * g.face_areas[bound_faces[hit]]

    xc = g.cell_centers
    source = rhs_f(xc[0], xc[1]) * g.cell_volumes

    perm = pp.SecondOrderTensor(g.dim, np.ones(g.num_cells))

    specified_parameters = {"second_order_tensor": perm, "source": source, "bc": bc, "bc_values": bc_val}
    data = pp.initialize_default_data(g, {}, kw_f, specified_parameters)

    A, b = tpfa.assemble_matrix_rhs(g, data)
    if dual_method:
        b[g.num_faces:] -= source
    else:
        b += source

    pressure = sps.linalg.spsolve(A, b)

    if dual_method:
        solution = pressure
        pressure = solution[g.num_faces:]

    u_ex = u_f(xc[0], xc[1])
    l2_real = np.sqrt(np.sum(g.cell_volumes * (pressure - u_ex)**2))

    print('Real error :' + str(l2_real))

    data[grid_variable] = pressure
    if not dual_method:
        pp.fvutils.compute_darcy_flux(g, data=data)
        flux = data[pp.DISCRETIZATION_MATRICES][kw_f]['flux'] * data[grid_variable] + \
                data[pp.DISCRETIZATION_MATRICES][kw_f]['bound_flux'] * data[pp.PARAMETERS][kw_f]['bc_values']
        data[pp.PARAMETERS][kw_f]['full_flux'] = flux
    else:
        num_flux = solution[:g.num_faces]
        data[pp.PARAMETERS][kw_f]['darcy_flux'] = num_flux
        data[pp.PARAMETERS][kw_f]['full_flux'] = num_flux

    xf_all = g.face_centers
    exact_flux = np.sum(np.vstack((-dux_f(xf_all[0], xf_all[1]), -duy_f(xf_all[0], xf_all[1]))) * g.face_normals[:2], axis=0)

    xn = g.nodes

    exact_p_nodes = u_f(xn[0], xn[1])


def compute_node_pressure(g, data, method=None):
    nc = g.num_cells
    nf = g.num_faces
    nn = g.num_nodes

    cell_nodes = g.cell_nodes()

    cell_node_volumes = cell_nodes * sps.dia_matrix((g.cell_volumes, 0), (nc, nc))

    sum_cell_nodes = cell_node_volumes * np.ones(nc)
    cell_nodes_scaled = sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes

    flux = data[pp.PARAMETERS][kw_f]['full_flux']
    proj_flux = pp.RT0(kw_f).project_flux(g, flux, data)[:g.dim]

    loc_grad = np.zeros((g.dim, nc))
    perm = data[pp.PARAMETERS][kw_f]['second_order_tensor'].values

    for ci in range(nc):
        loc_grad[:g.dim, ci] = -np.linalg.inv(perm[:g.dim, :g.dim, ci]).dot(proj_flux[:, ci])

    cell_node_matrix = g.cell_node_matrix()

    node_values = np.zeros(nn)

    for col in range(g.dim + 1):
        nodes = cell_node_matrix[:, col]
        dist = g.nodes[:g.dim, nodes] - g.cell_centers[:g.dim]

        scaling = cell_nodes_scaled[nodes, np.arange(nc)]
        contribution =  (np.asarray(scaling) \
            * (data[grid_variable] + np.sum(dist * loc_grad, axis=0))).ravel()
        node_values += np.bincount(nodes, weights=contribution, minlength=nn)


    bc = data[pp.PARAMETERS][kw_f]['bc']
    bc_values = data[pp.PARAMETERS][kw_f]['bc_values']

    external_dirichlet_boundary = np.logical_and(bc.is_dir, g.tags['domain_boundary_faces'])

    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = g.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_values[external_dirichlet_boundary]

    node_val_dir = g.face_nodes * face_vec

    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    node_values[is_dir_node] = node_val_dir[is_dir_node]

    return node_values



def reconstruct_flux(g, data):
    flux = data[pp.PARAMETERS][kw_f]['darcy_flux'].copy()

    for e, de in gb.edges_of_node(g):
        g_l = gb.nodes_of_edge(e)[0]
        if not g_l.dim < g.dim:
            continue
        mg = de['mortar_grid']

        flux -= mg.master_to_mortar_avg().T * de[mortar_variable]

    bc = data[pp.PARAMETERS][kw_f]['bc']
    bc_values = data[pp.PARAMETERS][kw_f]['bc_values']

    external_neumann_boundary = np.logical_and(bc.is_neu, g.tags['domain_boundary_faces']).sum()

    flux[external_neumann_boundary] = bc_values[external_neumann_boundary]
    return flux

flux = reconstruct_flux(g, data)

p_nodes = compute_node_pressure(g, data)

cn_mat = g.cell_node_matrix().T

p_diff = np.array([p_nodes[cn_mat[i]] - p_nodes[cn_mat[0]] for i in range(1, g.dim + 1)]).T

dx = np.empty(g.dim, dtype=np.object)
for dim in range(g.dim):
    dx[dim] = np.array([g.nodes[dim, cn_mat[i]] - g.nodes[dim, cn_mat[0]] for i in range(1, g.dim + 1)]).T


perm = data[pp.PARAMETERS][kw_f]['second_order_tensor'].values


if single_grid:
    # Use these lines for single grid
    if dual_method:
        flux = solution[:g.num_faces]
    else:
        flux = data[pp.DISCRETIZATION_MATRICES][kw_f]['flux'] * pressure + \
                data[pp.DISCRETIZATION_MATRICES][kw_f]['bound_flux'] * data[pp.PARAMETERS][kw_f]['bc_values']
    proj_flux = pp.RT0(kw_f).project_flux(g, flux, data)[:g.dim]
    data['projected_flux'] = proj_flux

    # Project the flux from faces to cells, using RT0 reconstruction.
    exact_projected_flux = pp.RT0(kw_f).project_flux(g, exact_flux, data)[:g.dim]

    # Gradient of a sort?
    exact_p_diff = np.array([exact_p_nodes[cn_mat[i]] - exact_p_nodes[cn_mat[0]] for i in range(1, g.dim + 1)]).T

else:
    pp.project_flux(gb, pp.RT0(kw_f), 'darcy_flux', 'projected_flux', mortar_variable)



nc = g.num_cells
sz = g.dim
mat = np.zeros((sz, sz))
rhs = np.zeros(sz)

kgrad = np.zeros((nc, sz))
kgrad_ex = 0 * kgrad

for ci in range(nc):
    for row in range(sz):
        mat[row] = dx[row][ci]
    mat = mat.T
    grad = np.linalg.solve(mat, p_diff[ci])

    kgrad[ci] = -perm[:sz, :sz, ci].dot(grad)

    if single_grid:
        grad_ex = np.linalg.solve(mat, exact_p_diff[ci])
        kgrad_ex[ci] = -perm[:sz, :sz, ci].dot(grad_ex)


kgrad = kgrad.T

proj_flux = data['projected_flux'][:2]
flux_diff = kgrad - proj_flux

L2_error = np.sqrt(np.sum(g.cell_volumes * np.sum(flux_diff**2, axis=0)))
print('Num cells: ' + str(g.num_cells) + ' L2 error: ' + str(L2_error))

if single_grid:
    flux_diff_ex = kgrad_ex.T - exact_projected_flux
    L2_error = np.sqrt(np.sum(g.cell_volumes * np.sum(flux_diff_ex**2, axis=0)))
    print('Num cells: ' + str(g.num_cells) + ' exact L2 error: ' + str(L2_error))

face_pressure = g.face_nodes.T * p_nodes / g.dim

cell_pressure = g.cell_nodes().T * p_nodes / (g.dim + 1)

exp = pp.Exporter(g, 'cell_errors')
if single_grid:
    exp.write_vtk({'error': np.sum(flux_diff**2, axis=0), 'pressure': data[grid_variable],
                   'exact_error':  np.sum(flux_diff_ex**2, axis=0), 'exact_pressure': u_ex})
else:
    exp.write_vtk({'error': np.sum(flux_diff**2, axis=0), 'pressure': data[grid_variable]})
#exp.write_vtk({'p': p_nodes}, point_data=True)
