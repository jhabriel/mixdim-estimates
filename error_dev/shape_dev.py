#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:51:09 2020

@author: jv
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps

# Utility function to obtain local nodes coordinates in 3D
def tri_to_list(p, tri):
    nt = tri.shape[1]
    return [np.array(p[:, tri[:, i]]) for i in range(nt)]

# domain -> unit square
corners = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])

# Generate three grids, using a bunch of random points + the corners
g4 = pp.TriangleGrid(np.hstack((np.random.rand(2, 200), corners)))
g4.compute_geometry()
p4 = tri_to_list(g4.nodes, g4.cell_nodes().indices.reshape((3, g4.num_cells), order='f'))

g5 = pp.TriangleGrid(np.hstack((np.random.rand(2, 100), corners)))
p5 = tri_to_list(g5.nodes, g5.cell_nodes().indices.reshape((3, g5.num_cells), order='f'))

g6 = pp.TriangleGrid(np.hstack((np.random.rand(2, 50), corners)))
p6 = tri_to_list(g6.nodes, g6.cell_nodes().indices.reshape((3, g6.num_cells), order='f'))

# Construct mappings
tri, mappings = pp.intersections.surface_tessalations([p4, p5, p6], return_simplexes=True)

# Check: The areas of the resulting triangulation should sum to 1
area = 0
for t in tri:
    v1 = t[:, 1] - t[:, 0]
    v2 = t[:, 2] - t[:, 0]
    area += 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])

print(area)

#%% 
list_of_grids = [g4, g5, g6]
for grid in list_of_grids:
    pp.plot_grid(grid, alpha=0.1, plot_2d=True)
    
#%%
# Mappings
g = g4.copy()
cell_nodes_map, _, _ = sps.find(g.cell_nodes())
cell_faces_map, _, _ = sps.find(g.cell_faces)
nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))
nodes_coor_cell = g4.nodes[:, nodes_cell]
