import numpy as np
import porepy as pp

# Disclaimer: Script copied or partially modified from 10.5281/zenodo.3374624


def create_grid(mesh_size, is_coarse, refine_1d, tol):
    # load the network
    file_name = "network_split.csv"
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # in the case of coarsened grid consider a first finer grid
    if is_coarse:
        if mesh_size == 0.06:
            mesh_size = 0.6 * 0.06
        elif mesh_size == 0.025:
            mesh_size = 0.7 * 0.025
        else:  # 0.0125
            mesh_size = 0.7 * 0.0125

    # create the mesh
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # assign the flag for the low permeable fractures
    gb = network.mesh(mesh_kwargs)

    # coarsened the grid
    if is_coarse:
        partition = pp.coarsening.create_aggregations(gb)
        partition = pp.coarsening.reorder_partition(partition)
        pp.coarsening.generate_coarse_grid(gb, partition)
    else:
        partition = None

    # refine the 1d grids
    if refine_1d:
        g_map = {}
        for g, _ in gb:
            if g.dim == 1:
                g_map[g] = pp.refinement.remesh_1d(g, g.num_nodes * 2)

        # gb = pp.mortars.replace_grids_in_bucket(gb, g_map, {}, tol)
        gb.replace_grids(g_map, {}, tol)
        gb.assign_node_ordering()

    # set the flag
    _set_flag(gb, tol)

    return gb, partition


def _set_flag(gb, tol):
    # set the key for the low peremable fractures
    gb.add_node_props("is_low")
    gb.add_node_props("frac_num")
    for g, d in gb:
        d["is_low"] = False
        d["frac_num"] = -1
        if g.dim == 1:

            f_0 = (g.nodes[0, :] - 0.05) / (0.2200 - 0.05) - (
                g.nodes[1, :] - 0.4160
            ) / (0.0624 - 0.4160)
            if np.sum(np.abs(f_0)) < tol:
                d["frac_num"] = 0

            f_1 = (g.nodes[0, :] - 0.05) / (0.2500 - 0.05) - (
                g.nodes[1, :] - 0.2750
            ) / (0.1350 - 0.2750)
            if np.sum(np.abs(f_1)) < tol:
                d["frac_num"] = 1

            f_2 = (g.nodes[0, :] - 0.15) / (0.4500 - 0.15) - (
                g.nodes[1, :] - 0.6300
            ) / (0.0900 - 0.6300)
            if np.sum(np.abs(f_2)) < tol:
                d["frac_num"] = 2

            f_3 = (g.nodes[0, :] - 0.15) / (0.4 - 0.15) - (g.nodes[1, :] - 0.9167) / (
                0.5 - 0.9167
            )
            if np.sum(np.abs(f_3)) < tol:
                d["frac_num"] = 3
                d["is_low"] = True

            f_4 = (g.nodes[0, :] - 0.65) / (0.849723 - 0.65) - (
                g.nodes[1, :] - 0.8333
            ) / (0.167625 - 0.8333)
            if np.sum(np.abs(f_4)) < tol:
                d["frac_num"] = 4
                d["is_low"] = True

            f_5 = (g.nodes[0, :] - 0.70) / (0.849723 - 0.70) - (
                g.nodes[1, :] - 0.2350
            ) / (0.167625 - 0.2350)
            if np.sum(np.abs(f_5)) < tol:
                d["frac_num"] = 5

            f_6 = (g.nodes[0, :] - 0.60) / (0.8500 - 0.60) - (
                g.nodes[1, :] - 0.3800
            ) / (0.2675 - 0.3800)
            if np.sum(np.abs(f_6)) < tol:
                d["frac_num"] = 6

            f_7 = (g.nodes[0, :] - 0.35) / (0.8000 - 0.35) - (
                g.nodes[1, :] - 0.9714
            ) / (0.7143 - 0.9714)
            if np.sum(np.abs(f_7)) < tol:
                d["frac_num"] = 7

            f_8 = (g.nodes[0, :] - 0.75) / (0.9500 - 0.75) - (
                g.nodes[1, :] - 0.9574
            ) / (0.8155 - 0.9574)
            if np.sum(np.abs(f_8)) < tol:
                d["frac_num"] = 8

            f_9 = (g.nodes[0, :] - 0.15) / (0.4000 - 0.15) - (
                g.nodes[1, :] - 0.8363
            ) / (0.9727 - 0.8363)
            if np.sum(np.abs(f_9)) < tol:
                d["frac_num"] = 9

    # we set know also the flag for the intersection, we need to go first through the
    # 0-dim grids and set there the is low and after to the edges
    gb.add_edge_props("is_low")
    for _, d in gb.edges():
        d["is_low"] = False

    for e, d in gb.edges():
        gl, gh = gb.nodes_of_edge(e)
        if gl.dim == 0 and gb.node_props(gh, "is_low"):
            gb.set_node_prop(gl, "is_low", True)

    # modify the key only for certain fractures
    for e, d in gb.edges():
        gl, gh = gb.nodes_of_edge(e)
        if gl.dim == 1 and gb.node_props(gl, "is_low"):
            d["is_low"] = True
