#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:22:09 2020

@author: jv
"""

import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

from mono_dim_fu import convergence_mono_grid

mesh_targets = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
methods = ['mpfa']
d = dict()
for method in methods:
    d[method] = {'mesh_size':[ ], 'error_DF':[ ], 'true_error':[ ], 'eff_idx':[ ]}
    for mesh_target in mesh_targets:
        h, e_true, e_DF, eff_idx = convergence_mono_grid(mesh_target)
        d[method]['mesh_size'].append(h)
        d[method]['error_DF'].append(e_DF)
        d[method]['true_error'].append(e_true)
        d[method]['eff_idx'].append(eff_idx)
  
#%% Fracture Pressure (L2-relative)
x1 = np.log(1e2)
y1 = np.log(5e-4)
y2 = np.log(1e-2)
x2 = x1 - y2 + y1

#plt.loglog([np.exp(x1), np.exp(x2)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

for method in methods:
    plt.loglog(1/np.array(d[method]['mesh_size']), 
                np.array(d[method]['error_DF']), marker='o')
    plt.loglog(1/np.array(d[method]['mesh_size']), 
                np.array(d[method]['true_error']), marker='*')

plt.legend(['Error Estimate', 'True Error'])
plt.xlabel('1/Mesh Size')
plt.ylabel('Fracture Error')
plt.title('Error in the fracture')
plt.grid(True)
plt.show()