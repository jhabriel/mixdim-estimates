#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:22:09 2020

@author: jv
"""

import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

from conv_fun import conv_fun

mesh_targets = np.array([0.1, 0.05 , 0.025, 0.0125])#, 0.00625])#, 0.003125])
methods = ['tpfa', 'mpfa', 'rt0', 'mvem']
methods = ['tpfa', 'mpfa']

d = dict()
for method in methods:
    #d[method] = {'mesh_size':[ ], 'error_bulk':[ ], 'error_frac':[ ], 'error_mortar':[ ], 'error_DF':[]}
    
    d[method] = {'num_cells_2d':[ ], 'true_error_2d':[ ], 'error_estimate_2d':[ ], 
                 'num_cells_1d':[ ], 'true_error_1d':[ ], 'error_estimate_1d':[ ],
                 'num_cells_mortar': [ ], "true_error_mortar": [ ], "error_estimate_mortar":  [] }
    
    for mesh_target in mesh_targets:
        (num_cells_2d, true_error_2d, error_estimate_2d,
         num_cells_1d, true_error_1d, error_estimate_1d) = conv_fun(mesh_target, method=method)
        
        d[method]['num_cells_2d'].append(num_cells_2d)
        d[method]['true_error_2d'].append(true_error_2d)
        d[method]['error_estimate_2d'].append(error_estimate_2d)
        
        d[method]['num_cells_1d'].append(num_cells_1d)
        d[method]['true_error_1d'].append(true_error_1d)
        d[method]['error_estimate_1d'].append(error_estimate_1d)
        
        d[method]['num_cells_mortar'].append(None)
        d[method]['true_error_mortar'].append(None)
        d[method]['error_estimate_mortar'].append(None)
        
        
#%% Bulk Pressure
plot_methods = ['tpfa', 'mpfa', 'rt0', 'mvem']
plot_methods = ['tpfa', 'mpfa']

plt.figure(0)
x1 = np.log(1e2)
y1 = np.log(2e-5)
y2 = np.log(1e-2)
x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
plt.grid(True, which="both", ls="-", color='0.35')
colors = ['red', 'blue', 'orange', 'green']

# Plot reference line
plt.loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

for counter, method in enumerate(plot_methods):
    plt.loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['true_error_2d']), color=colors[counter])
    plt.loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['error_estimate_2d']), marker='o', color=colors[counter])

plt.legend(['Quadratic', 'True TPFA', 'Estimate TPFA', 'True MPFA', 'Estimate MPFA',
            'True RT0', 'Estimate RT0', 'True MVEM', 'Estimate MVEM'])
plt.xlabel(r'$\sqrt{\# cells}$')
plt.ylabel(r'$error$')
plt.title('Convergence Analysis: Bulk Pressure')
plt.grid(True)
plt.show()

#%% Fracture Pressure
plt.figure(1)
x1 = np.log(1.5e2)
y1 = np.log(5e-11)
y2 = np.log(1e-7)
x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
plt.grid(True, which="both", ls="-", color='0.65')

# Plot reference line
plt.loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

for counter, method in enumerate(methods):
    plt.loglog(np.array(d[method]['num_cells_1d']), 
                np.array(d[method]['true_error_1d']), color=colors[counter])
    plt.loglog(np.array(d[method]['num_cells_1d']), 
                np.array(d[method]['error_estimate_1d']), marker='o', color=colors[counter])

plt.legend(['Quadratic', 'True TPFA', 'Estimate TPFA', 'True MPFA', 'Estimate MPFA',
            'True RT0', 'Estimate RT0', 'True MVEM', 'Estimate MVEM'])
plt.xlabel(r'$\# fracture \,\, cells$')
plt.ylabel(r'$error$')
plt.title('Convergence Analysis: Fracture Pressure')
plt.grid(True)
plt.show()

#%% Effectivity index 2D
plt.figure(2)
plot_methods = ['tpfa', 'mpfa', 'rt0', 'mvem']
plot_methods = ['tpfa', 'mpfa']

colors = ['red', 'blue', 'orange', 'green']

# Plot reference line

for counter, method in enumerate(plot_methods):
    plt.plot(np.array(d[method]['num_cells_2d']) ** 0.5, 
             np.array(d[method]['error_estimate_2d']) / np.array(d[method]['true_error_2d'] ), 
            marker='o', color=colors[counter])

plt.legend(['TPFA', 'MPFA', 'RT0', 'MVEM'])
plt.xlabel(r'$\sqrt{\# cells}$')
plt.ylabel(r'$effectivity \,\, index$')
plt.title('Convergence Analysis: Bulk Pressure')
plt.grid(True)
plt.show()

#%% Mortar Fluxes
#
# #%% Fracture Pressure (L2-relative)
# x1 = np.log(1e2)
# y1 = np.log(5e-4)
# y2 = np.log(1e-2)
# x2 = x1 - y2 + y1

# plt.loglog([np.exp(x1), np.exp(x2)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

# for method in methods:
#     plt.loglog(1/np.array(d[method]['mesh_size']), 
#                 np.array(d[method]['error_frac']), marker='o')

# plt.legend(['Linear'] + methods)
# plt.xlabel('1/Mesh Size')
# plt.ylabel('Fracture Error')
# plt.title('Error in the fracture')
# plt.grid(True)
# plt.show()

# #%% Mortar Flux (L2-relative)
# x1 = np.log(1e2)
# y1 = np.log(5e-4)
# y2 = np.log(1e-2)
# x2 = x1 - y2 + y1

# plt.loglog([np.exp(x1), np.exp(x2)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

# for method in methods:
#     plt.loglog(1/np.array(d[method]['mesh_size']), 
#                 np.array(d[method]['error_mortar']), marker='o')

# plt.legend(['Linear'] + methods)
# plt.xlabel('1/Mesh Size')
# plt.ylabel('Mortar Error')
# plt.title('Error in the mortar')
# plt.grid(True)
# plt.show()

# #%% Diffusive flux error
# x1 = np.log(1e2)
# y1 = np.log(5e-4)
# y2 = np.log(1e-2)
# x2 = x1 - y2 + y1

# plt.loglog([np.exp(x1), np.exp(x2)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

# for method in methods:
#     plt.loglog(1/np.array(d[method]['mesh_size']), 
#                 np.array(d[method]['error_DF']), marker='o')

# plt.legend(['Linear'] + methods)
# plt.xlabel('1/Mesh Size')
# plt.ylabel('Diffusive Flux Error')
# plt.title('A Posteriori Error Approximation')
# plt.grid(True)
# plt.show()