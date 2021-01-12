import os
import pathlib
from typing import List
import numpy as np
import porepy as pp
import scipy.sparse.linalg as spla


def setup_flow_assembler(gb, method, data_key=None, coupler=None):
    """ Setup a standard assembler for the flow problem for a given grid bucket.

    The assembler will be set up with primary variable name 'pressure' on the
    GridBucket nodes, and mortar_flux for the mortar variables.

    Parameters:
        gb: GridBucket.
        method (EllipticDiscretization).
        data_key (str, optional): Keyword used to identify data dictionary for
            node and edge discretization.
        Coupler (EllipticInterfaceLaw): Defaults to RobinCoulping.

    Returns:
        Assembler, ready to discretize and assemble problem.

    """

    if data_key is None:
        data_key = "flow"
    if coupler is None:
        coupler = pp.RobinCoupling(data_key, method)

    if isinstance(method, pp.MVEM) or isinstance(method, pp.RT0):
        mixed_form = True
    else:
        mixed_form = False

    for g, d in gb:
        if mixed_form:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1, "faces": 1}}
        else:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
        d[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {"mortar_flux": {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "lambda": {
                g1: ("pressure", "diffusive"),
                g2: ("pressure", "diffusive"),
                e: ("mortar_flux", coupler),
            }
        }
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    assembler = pp.Assembler(gb)

    num_blocks = assembler.full_dof.size
    block_info = np.zeros((num_blocks, 5))

    block_start = np.hstack((0, np.cumsum(assembler.full_dof)))

    # map from grids to block dof index. Will be unique, since there is a single
    # dof per subdomain
    subdom_block_map = {}

    for (g, var), ind in assembler.block_dof.items():
        is_mortar = 0

        if var == "mortar_flux":
            is_mortar = 1
            dim = g[0].dim
        else:
            dim = g.dim
            subdom_block_map[g] = ind

        block_info[ind, :3] = np.array([dim, is_mortar, block_start[ind]], dtype=np.int)

    # Second loop over the blocks. This time, we will fill in the two last
    # columns, on neighboring subdomains.
    for (g, var), ind in assembler.block_dof.items():
        if var == "mortar_flux":
            block_info[ind, 3] = subdom_block_map[g[0]]
            block_info[ind, 4] = subdom_block_map[g[1]]
        else:
            block_info[ind, 3:] = np.array([-1, -1])

    return assembler, block_info


def save_data(folder, extensions: List[str], matrices, rhs, data, matrix_format="coo"):

    fn = pathlib.Path("/home/eke001/Dropbox/bergen_tufts") / folder

    if not os.path.exists(fn):
        os.mkdir(fn)

    fn = str(fn)

    for ext, mat, b, d in zip(extensions, matrices, rhs, data):
        # Save data on matrix block structure
        np.savetxt(
            f"{fn}/{ext}_block_info.csv",
            d,
            delimiter=", ",
            header="DIMENSION, IS_MORTAR, START_OF_BLOCK, HIGH_DIM_NEIGH_VAR, LOW_DIM_NEIGH_VAR",
            fmt="%d",
        )

        np.savetxt(f"{fn}/{ext}_rhs", b)

        mat = mat.tocoo()
        mat_vals = np.vstack((mat.row, mat.col, mat.data)).T
        np.savetxt(f"{fn}/{ext}_matrix", mat_vals, fmt="%d %d %.18e", delimiter=", ")


def viz_gb(gb, folder, case):

    viz_folder = pathlib.Path(
        "/home/eke001/Dropbox/workspace/python/papers/xiaozhe_flow_preconditioner/viz"
    )
    full_folder = viz_folder / folder / case

    if not os.path.exists(folder):
        os.mkdir(folder)

    file_name = full_folder / "grid"

    exp = pp.Exporter(gb, str(file_name))
    exp.write_vtk()
    """
    template_file_name = "plot_grid_template.py"

    s = ""

    with open(viz_folder / template_file_name, "r") as f_in:
        s = f_in.readline()

    out_file_name = full_folder / "plot_grid.py"
    with open(out_file_name, "w") as f_out:
        f_out.write(s)
    """
