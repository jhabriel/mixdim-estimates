#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 03:32:13 2020

@author: jv
"""

import numpy as np
import scipy.sparse as sps
import porepy as pp

from a_posteriori_error import PosterioriError

#%% Define domain, fracture network, and construct grid bucket
# Define start and endpoints of the fractures.
pts = np.array([[0.25, 0.75], [0.25, 0.75]])
# Connection between the points (that is, the fractures) are specified as a 2 x num_frac array
connections = np.array([[0], [1]])
# Set the domain to the unit square, specified as a dictionary
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
# Define a 2d fracture network
network_2d = pp.FractureNetwork2d(pts, connections, domain)
# Plot fracture
network_2d.plot()
# Target lengths
target_h_bound = 0.1
target_h_fract = 0.05
mesh_args = {"mesh_size_bound": target_h_bound, "mesh_size_frac": target_h_fract}
# Construct grid bucket
gb = network_2d.mesh(mesh_args)
# Retrieve grids
g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)[0]
pp.plot_grid(g_2d, plot_2d=True)

#%%  Parameter assignment
# If you want to a setup which targets transport or mechanics problem,
# rather use the keyword 'transport' or 'mechanics'.
# For other usage, you will need to populate the parameter dictionary manually.
parameter_keyword = "flow"

# Maximum dimension of grids represented in the grid bucket
max_dim = gb.dim_max()

# Loop over all grids in the GridBucket.
# The loop will return a grid, and a dictionary used to store various data
for g, d in gb:

    # Permeability assignment
    # Differentiate between the rock matrix and the fractures.
    if g.dim == max_dim:
        kxx = np.ones(g.num_cells)
    else:  # g.dim == 1 or 0; note however that the permeability is not used in 0d domains
        kxx = 5 * np.ones(g.num_cells)

    perm = pp.SecondOrderTensor(kxx)

    # Create a dictionary to override the default parameters.
    # NB: The permeability is associated wiht the keyword second_order_tensor.
    specified_parameters = {"second_order_tensor": perm}

    # Add boundary conditions for 2d problems
    if g.dim == max_dim:
        # Dirichlet conditions on top and bottom
        # Note that the y-coordinates of the face centers are stored in the
        # second row (0-offset) of g.face_centers
        left = np.where(np.abs(g.face_centers[0]) < 1e-5)[0]
        right = np.where(np.abs(g.face_centers[0] - 1) < 1e-5)[0]

        # On the left and right boundaries, we set homogeneous Neumann conditions
        # Neumann conditions are set by default, so there is no need to do anything

        # Define BoundaryCondition object
        bc_faces = np.hstack((left, right))
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

        # Register the assigned value
        specified_parameters["bc"] = bc

        # Alse set the values - specified as vector of size g.num_faces
        bc_values = np.zeros(g.num_faces)
        bc_values[left] = 1
        bc_values[right] = 0
        specified_parameters["bc_values"] = bc_values

    # On 1d and 0d problems we set no boundary condition - in effect assigning Neumann conditions

    # Assign the values to the data dictionary d.
    # By using the method initialize_default_data, various other fields are also
    # added, see
    pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

    # Internally to the Parameter class, the parameters are stored as dictionaries.
    # To illustrate how to access specific sets of parameters, print the keywords
    # for one of the grids.
    # Note the nested dictionaries.
    if g.dim == max_dim:
        print("The assigned parameters for the 2d grid are")
        print(d[pp.PARAMETERS][parameter_keyword].keys())

# Next loop over the edges (interfaces) in the
for e, d in gb.edges():
    # On edges in the GridBucket, there is currently no methods for default initialization.

    # Set the normal diffusivity parameter (the permeability-like transfer coefficient)
    data = {"normal_diffusivity": 1}

    # Add parameters: We again use keywords to identify sets of parameters.
    #
    mg = d["mortar_grid"]
    pp.initialize_data(mg, d, parameter_keyword, data)


#%% Primary variables and discretizations
# We will use a multi-point flux approximation method on all subdomains
# Note that we need to provide the same keyword that we used when setting the parameters on the subdomain.
# If we change to pp.Mpfa(keyword='foo'), the discretization will not get access to the parameters it needs
subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)

# On all subdomains, variables are identified by a string.
# This need not be the same on all subdomains, even if the governing equations are the same,
# but here we go for the simple option.
subdomain_variable = "pressure"

# Identifier of the discretization operator for each term.
# In this case, this is seemingly too complex, but for, say, an advection-diffusion problem
# we would need two discretizaiton objects (advection, diffusion) and one keyword
# per operator
subdomain_operator_keyword = "diffusion"

# Specify discertization objects on the interfaces / edges
# Again give the parameter keyword, and also the discretizations used on the two neighboring
# subdomains (this is needed to discretize the coupling terms)
edge_discretization = pp.RobinCoupling(
    parameter_keyword, subdomain_discretization, subdomain_discretization
)
# Variable name for the interface variable
edge_variable = "interface_flux"
# ... and we need a name for the discretization opertaor for each coupling term
coupling_operator_keyword = "interface_diffusion"

# Loop over all subdomains in the GridBucket, assign parameters
# Note that the data is stored in sub-dictionaries
for g, d in gb:
    # Assign primary variables on this grid, compatible with the designated discretization scheme
    # In this case, the discretization has one degree of freedom per cell.
    # If we changed to a mixed finite element method, this could be {'cells': 1, "faces": 1}
    d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
    # Assign discretization operator for the variable.
    # If the discretization is composed of several terms, they can be assigned
    # by multiple entries in the inner dictionary, e.g.
    #  {operator_keyword_1: method_1, operator_keyword_2: method_2, ...}
    d[pp.DISCRETIZATION] = {
        subdomain_variable: {subdomain_operator_keyword: subdomain_discretization}
    }

# Next, loop over the edges
for e, d in gb.edges():
    # Get the grids of the neighboring subdomains
    # The first will always be the lower-dimensional
    g1, g2 = gb.nodes_of_edge(e)
    # The interface variable has one degree of freedom per cell in the mortar grid
    # This is essentially a DG(0) discretization
    d[pp.PRIMARY_VARIABLES] = {edge_variable: {"cells": 1}}

    # The coupling discretization links an edge discretization with variables
    # and discretization operators on each neighboring grid
    d[pp.COUPLING_DISCRETIZATION] = {
        # This assignment associate this coupling term with a unique combination of variable
        # and term (thus discretization object) applied to each of the neighboring subdomains.
        # Again, the complexity is not warranted for this problem, but necessary for general
        # multi-physics problems with non-trivial couplings between variables.
        coupling_operator_keyword: {
            g1: (subdomain_variable, subdomain_operator_keyword),
            g2: (subdomain_variable, subdomain_operator_keyword),
            e: (edge_variable, edge_discretization),
        }
    }

#%% Assembly and solve
# Initialize the assembler with the GridBucket it should operate on.
# Initialization will process the variables defined above (name, number, and number of dofs)
# and assign global degrees of freedom.
assembler = pp.Assembler(gb)

# First discretize. This will call all discretizaiton operations defined above
assembler.discretize()

# Finally, assemble matrix and right hand side
A, b = assembler.assemble_matrix_rhs()

# Direct solver
sol = sps.linalg.spsolve(A, b)

# The solution vector is a global vector. Distribute it to the local grids and interfaces
assembler.distribute_variable(sol)

#%% Obtaining the errors
# The errors are stored in the dictionaries under pp.STATE
PosterioriError(
    gb, parameter_keyword, subdomain_variable, nodal_method="k-averaging", p_order="1"
)

#%% Write to vtk. Create a new exporter, to avoid interferring with the above grid illustration.
exporter = pp.Exporter(gb, "flow", folder_name="md_flow")
exporter.write_vtk()
# Note that we only export the subdomain variable, not the one on the edge
exporter.write_vtk([subdomain_variable, "error_DF"])
