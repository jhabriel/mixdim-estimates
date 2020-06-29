#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:22:09 2020

@author: jv
"""

import numpy as np
import os, sys
import seaborn as sns
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

#from conv_fun import conv_fun
from conf_fun_new import conv_fun_new as conv_fun

mesh_targets = np.array([0.1, 0.05 , 0.025, 0.0125])
methods = ['mvem', 'rt0', 'tpfa', 'mpfa']
#methods  = ['mpfa']
#methods = ['rt0']

d = dict()
for method in methods:
    #d[method] = {'mesh_size':[ ], 'error_bulk':[ ], 'error_frac':[ ], 'error_mortar':[ ], 'error_DF':[]}
    
    d[method] = {'num_cells_2d':[ ], 'true_error_2d':[ ], 'error_estimate_2d':[ ], 
                 'num_cells_1d':[ ], 'true_error_1d':[ ], 'error_estimate_1d':[ ],
                 'num_cells_mortar': [ ], "true_error_mortar": [ ], "error_estimate_mortar":  [], 
                 'l2_velocity_2d': [ ], "l2_velocity_1d": [ ],
                 'l2_postp_2d' : [ ], 'l2_postp_1d': [ ],
                 'l2_recons_2d': [ ], 'l2_recons_1d': [ ],
                 'l2_direct_recons_2d': [ ], 'l2_direct_recons_1d': [ ],
                 'direct_ee_2d': [ ],   'direct_ee_1d': [ ],
                 #"new_sh_1d": [ ]
                 #'l2_conf_nv_2d': [ ], "l2_conf_nv_1d": [ ], 
                 #'l2_grad_nonconf_2d': [ ], "l2_grad_nonconf_1d": [ ], 
                 }
    
    for mesh_target in mesh_targets:
        (num_cells_2d, true_error_2d, error_estimate_2d,
         num_cells_1d, true_error_1d, error_estimate_1d,
         num_cells_mortar, true_error_mortar, error_estimates_mortar,
         l2_velocity_2d, l2_velocity_1d,
         l2_postp_2d,    l2_postp_1d, 
         l2_recons_2d,   l2_recons_1d,
         l2_direct_recons_2d, l2_direct_recons_1d,
         direct_ee_2d, direct_ee_1d,
         #new_sh_1d
         #l2_conf_nv_2d, l2_conf_nv_1d, 
         #l2_grad_nonconf_2d, l2_grad_nonconf_1d
         ) = conv_fun(mesh_target, method=method)
        
        d[method]['num_cells_2d'].append(num_cells_2d)
        d[method]['true_error_2d'].append(true_error_2d)
        d[method]['error_estimate_2d'].append(error_estimate_2d)
        
        d[method]['num_cells_1d'].append(num_cells_1d)
        d[method]['true_error_1d'].append(true_error_1d)
        d[method]['error_estimate_1d'].append(error_estimate_1d)
        
        d[method]['num_cells_mortar'].append(num_cells_mortar)
        d[method]['true_error_mortar'].append(true_error_mortar)
        d[method]['error_estimate_mortar'].append(error_estimates_mortar)
        
        d[method]['l2_velocity_2d'].append(l2_velocity_2d)
        d[method]['l2_velocity_1d'].append(l2_velocity_1d)
        d[method]['l2_postp_2d'].append(l2_postp_2d)
        d[method]['l2_postp_1d'].append(l2_postp_1d)
        d[method]['l2_recons_2d'].append(l2_recons_2d)
        d[method]['l2_recons_1d'].append(l2_recons_1d)
        d[method]['l2_direct_recons_2d'].append(l2_direct_recons_2d)
        d[method]['l2_direct_recons_1d'].append(l2_direct_recons_1d)
        d[method]['direct_ee_2d'].append(direct_ee_2d)
        d[method]['direct_ee_1d'].append(direct_ee_1d)
        #d[method]['new_sh_1d'].append(new_sh_1d)

        #d[method]['l2_conf_nv_2d'].append(l2_conf_nv_2d)
        #d[method]['l2_conf_nv_1d'].append(l2_conf_nv_1d)
        #d[method]['l2_grad_nonconf_2d'].append(l2_grad_nonconf_2d)
        #d[method]['l2_grad_nonconf_1d'].append(l2_grad_nonconf_1d)

#%%
# Colors specification
if len(methods) == 1:
    colors = ['red']
elif len(methods) == 2:
    colors = ['red', 'blue']
elif len(methods) == 3:
    colors = ['red', 'blue', 'orange']
else:      
    colors = ['red', 'orange', 'green', 'blue']


#%% Create empty subplot
def create_subplot(
    fig_num,
    sns_context="paper",
    sns_color_palette="tab10",
    sns_max_color_num=10,
    sns_style="whitegrid",
):

    sns.set_context(sns_context)  # set scale and size of figures
    sns.set_palette(sns_color_palette, sns_max_color_num)
    itertools.cycle(sns.color_palette())  # iterate if > 10 colors are needed

    fig = plt.figure(fig_num, constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.01, right=0.40)
    gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.54, right=0.93)
    gs3 = fig.add_gridspec(nrows=1, ncols=1, left=0.95, right=0.99)

    # Assingning one frame to each plot
    with sns.axes_style(sns_style):  # assign the style
        ax1 = fig.add_subplot(gs1[0, 0])
        ax2 = fig.add_subplot(gs2[0, 0])

    ax3 = fig.add_subplot(gs3[0, 0])

    axes = [ax1, ax2, ax3]

    return fig, axes


#%% Primal norm
figure, axes = create_subplot(0)

# 2d plot
#ax1 = plt.subplot(131)
x1 = np.log(2.7e2)
y1 = np.log(2e-5)
y2 = np.log(1e-2)
x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[0].grid(True, which="both", ls="--", color='0.75')

# Plot reference line
figure.axes[0].loglog(
    [np.exp(x1), np.exp(x2_quadra)], 
    [np.exp(y1), np.exp(y2)], 
    color='k', LineWidth=3
)

for counter, method in enumerate(methods):
    figure.axes[0].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['true_error_2d']), linestyle='--', linewidth=2, color=colors[counter])
    figure.axes[0].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['error_estimate_2d']), marker='o', linewidth=2, markersize=5, color=colors[counter])
    #figure.axes[0].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
    #            np.array(d[method]['direct_ee_2d']), marker='o', color='blue')

figure.axes[0].set_xlabel(r'$\sqrt{\# cells}$')
figure.axes[0].set_ylabel(r'$\eta_{NC}^2 \;\; (\Omega_2)$')
figure.axes[0].grid(True)

# 1d plot
#ax2 = plt.subplot(132)
x1 = np.log(5e1)
y1 = np.log(2.5e-8)
y2 = np.log(3e-5)
x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[1].grid(True, which="both", ls="--", color='0.75')

# Plot reference line
figure.axes[1].loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

for counter, method in enumerate(methods):
    figure.axes[1].loglog(np.array(d[method]['num_cells_1d']), 
                np.array(d[method]['true_error_1d']), linestyle='--', linewidth=2, color=colors[counter])
    figure.axes[1].loglog(np.array(d[method]['num_cells_1d']), 
                np.array(d[method]['direct_ee_1d']), marker='o', color=colors[counter])
    figure.axes[1].set_xlabel(r'$\# fracture \,\, cells$')
figure.axes[1].set_ylabel(r'$\eta_{NC}^2 \;\; (\Omega_1)$')
figure.axes[1].grid(True)

# Legends
#ax3 = plt.subplot(133)
figure.axes[2].loglog([],[], color='k', LineWidth=3)
for counter, method in enumerate(methods):
    figure.axes[2].loglog([],[], linewidth=2, linestyle='--',color=colors[counter])
    figure.axes[2].loglog([],[], linewidth=2, markersize=6, marker='o', color=colors[counter])
    #figure.axes[2].loglog([],[], marker='o', color='blue')
#str_legend = ["Quadratic"]
#for method in methods: 
#    str_legend.append('True ' + method)
#    str_legend.append('Estimate ' + method)
str_legend = ["Quadratic", 
              'True error MVEM', 'Estimate MVEM', 'True error RT0', 'Estimate RT0',
              'True error TPFA', 'Estimate TPFA', 'True error MPFA', 'Estimate MPFA']
#str_legend.append('Estimate direct ' + method)
figure.axes[2].legend(str_legend)
figure.axes[2].legend(str_legend, loc="center left", frameon=False)
figure.axes[2].axis("off")

figure.suptitle("Convergence analysis for bulk and fracture")

# #%% Energy norm - Mortar
# plt.figure(2)
# x1 = np.log(1.5e2)
# y1 = np.log(5e-11)
# y2 = np.log(1e-7)

# x2_linear = x1 - (y2 - y1)
# x2_quadra = x1 - 0.5*(y2-y1) 
# plt.grid(True, which="both", ls="-", color='0.35')

# # Plot reference line
# #plt.loglog([np.exp(x1), np.exp(x2_linear)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)
# plt.loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

# for counter, method in enumerate(methods):
#     plt.loglog(np.array(d[method]['num_cells_mortar']), 
#                 np.array(d[method]['true_error_mortar']), color=colors[counter])
#     plt.loglog(np.array(d[method]['num_cells_1d']), 
#                 np.array(d[method]['error_estimate_mortar']), marker='o', color=colors[counter])

# plt.legend(['Linear','Quadratic', 'True TPFA', 'Estimate TPFA', 'True MPFA', 'Estimate MPFA',
#             'True RT0', 'Estimate RT0', 'True MVEM', 'Estimate MVEM'])
# plt.xlabel(r'$\# fracture \,\, cells$')
# plt.ylabel(r'$error$')
# plt.title('Convergence Analysis: Mortar fluxes')
# plt.grid(True)
# plt.show()

#%% Reconstructed velocity
figure, axes = create_subplot(1)

x1 = np.log(1.5e2)
y1 = np.log(0.5e-4)
y2 = np.log(2e-2)

x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[0].grid(True, which="both", ls="--", color='0.75')

# Plot 2d
figure.axes[0].loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

for counter, method in enumerate(methods):
    figure.axes[0].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['l2_velocity_2d']), marker='o', color=colors[counter])    

figure.axes[0].set_xlabel(r'$\sqrt{\# cells}$')
figure.axes[0].set_ylabel(r'$error \;\; (\Omega_2)$')

# Plot 1d

x1 = np.log(3e1)
y1 = np.log(4e-7)
y2 = np.log(4e-5)

x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[1].grid(True, which="both", ls="--", color='0.75')

figure.axes[1].loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

for counter, method in enumerate(methods):
    figure.axes[1].loglog(np.array(d[method]['num_cells_1d']), 
                np.array(d[method]['l2_velocity_1d']), marker='o', color=colors[counter])    
    
figure.axes[1].set_xlabel(r'$\# fracture \,\, cells$')
figure.axes[1].set_ylabel(r'$error \;\; (\Omega_1)$')

# Plot legend
figure.axes[2].loglog([],[], color='m', LineWidth=3)
for counter, method in enumerate(methods):
    figure.axes[2].loglog([],[], marker='o', color=colors[counter])      
leg_str = ['Quadratic']
for method in methods: leg_str.append(method) 
figure.axes[2].legend(leg_str, loc="center left", frameon=False)
figure.axes[2].axis("off")

figure.suptitle(r"$||u - \tilde{u}_h|| $")


#%% Postprocessed pressure

figure, axes = create_subplot(2)

x1 = np.log(1.5e2)
y1 = np.log(3e-7)
y2 = np.log(1e-4)

x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[0].grid(True, which="both", ls="--", color='0.75')

# Plot 2d
figure.axes[0].loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

for counter, method in enumerate(methods):
    figure.axes[0].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['l2_postp_2d']), marker='o', color=colors[counter])    

figure.axes[0].set_xlabel(r'$\sqrt{\# cells}$')
figure.axes[0].set_ylabel(r'$error \;\; (\Omega_2)$')

# Plot 1d

x1 = np.log(4e1)
y1 = np.log(7e-10)
y2 = np.log(3e-7)

x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[1].grid(True, which="both", ls="--", color='0.75')

figure.axes[1].loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

for counter, method in enumerate(methods):
    figure.axes[1].loglog(np.array(d[method]['num_cells_1d']), 
                np.array(d[method]['l2_postp_1d']), marker='o', color=colors[counter])    
    
figure.axes[1].set_xlabel(r'$\# fracture \,\, cells$')
figure.axes[1].set_ylabel(r'$error \;\; (\Omega_1)$')

# Plot legend
figure.axes[2].loglog([],[], color='m', LineWidth=3)
for counter, method in enumerate(methods):
    figure.axes[2].loglog([],[], marker='o', color=colors[counter])      
leg_str = ['Quadratic']
for method in methods: leg_str.append(method) 
figure.axes[2].legend(leg_str, loc="center left", frameon=False)
figure.axes[2].axis("off")

figure.suptitle(r"$||p - \tilde{p}_h|| $")


#%% Reconstructed (conforming) pressure

figure, axes = create_subplot(3)

x1 = np.log(1.5e2)
y1 = np.log(3e-7)
y2 = np.log(1e-4)

x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[0].grid(True, which="both", ls="--", color='0.75')

# Plot 2d
figure.axes[0].loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

for counter, method in enumerate(methods):
    figure.axes[0].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['l2_recons_2d']), marker='o', color=colors[counter])
    figure.axes[0].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['l2_direct_recons_2d']), marker='s', color='blue')

figure.axes[0].set_xlabel(r'$\sqrt{\# cells}$')
figure.axes[0].set_ylabel(r'$error \;\; (\Omega_2)$')

# Plot 1d

x1 = np.log(4e1)
y1 = np.log(7e-10)
y2 = np.log(3e-7)

x2_linear = x1 - (y2 - y1)
x2_quadra = x1 - 0.5*(y2-y1) 
figure.axes[1].grid(True, which="both", ls="--", color='0.75')

figure.axes[1].loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

for counter, method in enumerate(methods):
    figure.axes[1].loglog(np.array(d[method]['num_cells_1d']), 
                np.array(d[method]['new_sh_1d']), marker='o', color=colors[counter])    
    figure.axes[1].loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
                np.array(d[method]['l2_direct_recons_1d']), marker='s', color='blue')
    
figure.axes[1].set_xlabel(r'$\# fracture \,\, cells$')
figure.axes[1].set_ylabel(r'$error \;\; (\Omega_1)$')

# Plot legend
figure.axes[2].loglog([],[], color='m', LineWidth=3)
for counter, method in enumerate(methods):
    figure.axes[2].loglog([],[], marker='o', color=colors[counter])    
    figure.axes[2].loglog([],[], marker='o', color='blue')   
leg_str = ['Quadratic', 'Vohralik', 'Direct reconst.']
#for method in methods: leg_str.append(method) 
figure.axes[2].legend(leg_str, loc="center left", frameon=False)
figure.axes[2].axis("off")

figure.suptitle(r"$\eta = \sum_{K \in T_h} ||p - s_h||_K^2$")


#%%
# # -------------------------------------------------------------------------- #

# plt.figure(5)
# x1 = np.log(3e2)
# y1 = np.log(8e-4)
# y2 = np.log(2e-2)

# x2_linear = x1 - (y2 - y1)
# x2_quadra = x1 - 0.5*(y2-y1) 
# plt.grid(True, which="both", ls="-", color='0.35')

# # Plot reference line
# plt.loglog([np.exp(x1), np.exp(x2_linear)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

# for counter, method in enumerate(methods):
#     plt.loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
#                 np.array(d[method]['l2_conf_nv_2d']), marker='o', color=colors[counter])    
    
# leg_str = ['Linear']
# for method in methods: leg_str.append('L2-'+method) 
# plt.legend(leg_str)
# plt.xlabel(r'$\sqrt{\# cells}$')
# plt.ylabel(r'$error$')
# plt.title('L2  - Conforming Nodal Pressure (2D)')
# plt.grid(True)
# plt.show()

# # -------------------------------------------------------------------------- #

# plt.figure(6)
# x1 = np.log(8e1)
# y1 = np.log(5e-5)
# y2 = np.log(8e-4)

# x2_linear = x1 - (y2 - y1)
# x2_quadra = x1 - 0.5*(y2-y1) 
# plt.grid(True, which="both", ls="-", color='0.35')

# # Plot reference line
# plt.loglog([np.exp(x1), np.exp(x2_linear)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)
# #plt.loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

# for counter, method in enumerate(methods):
#     plt.loglog(np.array(d[method]['num_cells_1d']), 
#                 np.array(d[method]['l2_conf_nv_1d']), marker='o', color=colors[counter])    
    
# leg_str = ['Linear']
# for method in methods: leg_str.append('L2-'+method) 
# plt.legend(leg_str)
# plt.xlabel(r'$\# fracture \,\, cells$')
# plt.ylabel(r'$error$')
# plt.title('L2 - Conforming Nodal Pressure (1D)')
# plt.grid(True)
# plt.show()

# # -------------------------------------------------------------------------- #

# plt.figure(7)
# x1 = np.log(3e2)
# y1 = np.log(8e-4)
# y2 = np.log(2e-2)

# x2_linear = x1 - (y2 - y1)
# x2_quadra = x1 - 0.5*(y2-y1) 
# plt.grid(True, which="both", ls="-", color='0.35')

# # Plot reference line
# plt.loglog([np.exp(x1), np.exp(x2_linear)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)

# for counter, method in enumerate(methods):
#     plt.loglog(np.array(d[method]['num_cells_2d']) ** 0.5, 
#                 np.array(d[method]['l2_grad_nonconf_2d']), marker='o', color=colors[counter])    
    
# leg_str = ['Linear']
# for method in methods: leg_str.append('L2-'+method) 
# plt.legend(leg_str)
# plt.xlabel(r'$\sqrt{\# cells}$')
# plt.ylabel(r'$error$')
# plt.title('L2 - Full fluxes (2D)')
# plt.grid(True)
# plt.show()

# # -------------------------------------------------------------------------- #

# plt.figure(8)
# x1 = np.log(8e1)
# y1 = np.log(5e-5)
# y2 = np.log(8e-4)

# x2_linear = x1 - (y2 - y1)
# x2_quadra = x1 - 0.5*(y2-y1) 
# plt.grid(True, which="both", ls="-", color='0.35')

# # Plot reference line
# plt.loglog([np.exp(x1), np.exp(x2_linear)], [np.exp(y1), np.exp(y2)], color='m', LineWidth=3)
# #plt.loglog([np.exp(x1), np.exp(x2_quadra)], [np.exp(y1), np.exp(y2)], color='k', LineWidth=3)

# for counter, method in enumerate(methods):
#     plt.loglog(np.array(d[method]['num_cells_1d']), 
#                 np.array(d[method]['l2_grad_nonconf_1d']), marker='o', color=colors[counter])    

# leg_str = ['Linear']
# for method in methods: leg_str.append('L2-'+method) 
# plt.legend(leg_str)
# plt.xlabel(r'$\# fracture \,\, cells$')
# plt.ylabel(r'$error$')
# plt.title('L2 - Full fluxes (1D)')
# plt.grid(True)
# plt.show()

#%% Effectivity index 2D
# plt.figure(2)
# plot_methods = ['tpfa', 'mpfa', 'rt0', 'mvem']
# plot_methods = ['tpfa', 'mpfa']

# colors = ['red', 'blue', 'orange', 'green']

# # Plot reference line

# for counter, method in enumerate(plot_methods):
#     plt.plot(np.array(d[method]['num_cells_2d']) ** 0.5, 
#              np.array(d[method]['error_estimate_2d']) / np.array(d[method]['true_error_2d'] ), 
#             marker='o', color=colors[counter])

# plt.legend(['TPFA', 'MPFA', 'RT0', 'MVEM'])
# plt.xlabel(r'$\sqrt{\# cells}$')
# plt.ylabel(r'$effectivity \,\, index$')
# plt.title('Convergence Analysis: Bulk Pressure')
# plt.grid(True)
# plt.show()

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