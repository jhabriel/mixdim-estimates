import porepy as pp
import numpy as np

from analytical import ExactSolution

#%% Create a grid
mesh_size = 0.1
domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
network_2d = pp.FractureNetwork2d(None, None, domain)
mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}

gb = network_2d.mesh(mesh_args)
g = gb.grids_of_dimension(2)[0]
#pp.plot_grid(g, alpha=0.1, plot_2d=True, figsize=(5, 5))

#%%
