#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:05:25 2020

@author: jv

Testing of a single 3D-2D fracture network with non-matching grids
"""

import porepy as pp
import numpy as np
import scipy.sparse as sps

#%% Utility functions

def make_gb(mesh_size=0.3, mesh_size_min=0.2, export_to_vtk=False):
    
    f = pp.Fracture(np.array([[0.5, 0.5, 0.5, 0.5], [-1, 2, 2, -1], [-1, -1, 2, 2]]))
    domain = {'xmin': -2, 'xmax': 3, 'ymin': -2, 'ymax': 3, 'zmin': -3, 'zmax': 3}
    network = pp.FractureNetwork3d(f, domain=domain)
    
    mesh_args = {
        "mesh_size_bound": mesh_size, 
        "mesh_size_frac": mesh_size,
        "mesh_size_min": mesh_size_min}
    gb = network.mesh(mesh_args, ensure_matching_face_cell=False)
    
    if export_to_vtk:
        network.to_vtk('fracture_network.vtu')
        
    return gb

#%% Create three grid buckets with different refinement levels
h0 = 0.3 # initial mesh size
r = 1.5  # refinement ratio 

#%% Create grid buckets with differnt levels of refinement
gb1 = make_gb(mesh_size=h0)
gb2 = make_gb(mesh_size=h0/r)
gb3 = make_gb(mesh_size=h0/(2*r))

#%% Unify grid buckets into the coarsest one
gb = gb1.copy()
g_map  = {gb.grids_of_dimension(2)[0]: gb3.grids_of_dimension(2)[0]}
mg_map = {gb.get_mortar_grids()[0]: gb2.get_mortar_grids()[0]}
gb.replace_grids(g_map=g_map, mg_map=mg_map)

