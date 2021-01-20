import numpy as np
import porepy as pp
import itertools
import mdestimates as mde

from time import time
from data import add_data
from create_grid import create_grid
from solvers import run_flow


def homo_rt0(g):
    return {
        "scheme": pp.RT0("flow"),
        "dof": {"cells": 1, "faces": 1},
        "label": "homo_rt0",
    }


def homo_tpfa(g):
    return {"scheme": pp.Tpfa("flow"), "dof": {"cells": 1}, "label": "homo_tpfa"}


def homo_mpfa(g):
    return {"scheme": pp.Mpfa("flow"), "dof": {"cells": 1}, "label": "homo_mpfa"}


def homo_mvem(g):
    return {
        "scheme": pp.MVEM("flow"),
        "dof": {"cells": 1, "faces": 1},
        "label": "homo_mvem",
    }


def hete1(g):
    if g.dim == 2:
        scheme = {
            "scheme": pp.RT0("flow"),
            "dof": {"cells": 1, "faces": 1},
            "label": "hete1",
        }
    else:
        scheme = {"scheme": pp.Tpfa("flow"), "dof": {"cells": 1}, "label": "hete1"}
    return scheme


def hete2(g):
    if g.dim == 2:
        scheme = {
            "scheme": pp.MVEM("flow"),
            "dof": {"cells": 1, "faces": 1},
            "label": "hete2",
        }
    else:
        scheme = {"scheme": pp.Tpfa("flow"), "dof": {"cells": 1}, "label": "hete2"}
    return scheme


def homo_mortar(g):
    return {
        "scheme": pp.RT0("flow"),
        "dof": {"cells": 1, "faces": 1},
        "label": "homo_mortar",
    }


def main(mesh_size, discr, flow_dir, is_coarse, refine_1d, folder):

    # set the geometrical tolerance
    tol = 1e-6
    # create the gb
    gb, partition = create_grid(mesh_size, is_coarse, refine_1d, tol)
    # set the scheme for each grid
    for g, d in gb:
        d["discr"] = discr(g)
    # add the problem data
    add_data(gb, flow_dir, tol)
    # solve the darcy problem
    run_flow(gb, partition, folder)

    return gb


if __name__ == "__main__":

    # Numerical methods
    # NOTE: The second parameter is requested in case of coasened grid, only
    # when MVEM is applied to the 2d. The third parameter is related to
    # the mortar
    solver_list = {
        "tpfa": (homo_tpfa, False, False),
        "mpfa": (homo_mpfa, False, False),
        "rt0": (homo_rt0, False, False),
        "mvem": (homo_mvem, False, False),
    }
    numerical_methods = list(solver_list.keys())

    # Flow directions
    flow_dirs = ["left_to_right"]
    # Mesh sizes
    mesh_sizes = [0.05, 0.025, 0.0125]
    mesh_resolutions = []
    for mesh_size in mesh_sizes:
        if mesh_size == 0.05:
            mesh_resolutions.append("coarse")
        elif mesh_size == 0.025:
            mesh_resolutions.append("intermediate")
        elif mesh_size == 0.0125:
            mesh_resolutions.append("fine")
        else:
            raise ValueError("Mesh size not part of the benchmark study.")

    # Initialize dictionary to store the errors
    out = {k: {} for k in numerical_methods}
    for i in itertools.product(numerical_methods, mesh_resolutions):
        out[i[0]][i[1]] = {
            "error_node_2d": [],
            "error_node_1d_cond": [],
            "error_node_1d_bloc": [],
            "error_edge_1d_cond": [],
            "error_edge_1d_bloc": [],
            "error_edge_0d": [],
            "error_global_scaled": [],
            "error_global": [],
        }

    # Loop and solve
    for solver_name, (solver, is_coarse, refine_1d) in solver_list.items():
        for flow_dir in flow_dirs:
            for (mesh_size, resol) in zip(mesh_sizes, mesh_resolutions):
                # Get folder for exporting to PARAVIEW
                folder = solver_name + "_" + flow_dir + "_" + str(mesh_size)

                # Solve for a given method and a given mesh size
                print("Solving with", solver_name, "for mesh size", mesh_size)
                tic = time()
                gb = main(mesh_size, solver, flow_dir, is_coarse, refine_1d, folder)
                print(f"Problem succesfully solved. Time {time() - tic}")

                # Estimate errors
                tic = time()
                estimates = mde.ErrorEstimate(gb, lam_name="mortar_flux")
                estimates.estimate_error()
                estimates.transfer_error_to_state()
                scaled_majorant = estimates.get_scaled_majorant()

                print(f"Errors succesfully estimated. Time {time() - tic}")

                # Compute errors
                error_node_2d = 0
                error_node_1d_cond = 0
                error_node_1d_bloc = 0
                error_edge_1d_cond = 0
                error_edge_1d_bloc = 0
                error_edge_0d = 0

                # Get subdomain errors
                for g, d in gb:
                    if g.dim == 2:
                        error_node_2d += estimates.get_scaled_local_errors(g, d)
                    elif g.dim == 1:
                        if d["is_low"]:
                            error_node_1d_bloc += estimates.get_scaled_local_errors(
                                g, d
                            )
                        else:
                            error_node_1d_cond += estimates.get_scaled_local_errors(
                                g, d
                            )
                    else:
                        continue

                # Get interface errors
                for e, d in gb.edges():
                    mg = d["mortar_grid"]
                    if mg.dim == 1:
                        if d["is_low"]:
                            error_edge_1d_bloc += estimates.get_scaled_local_errors(
                                mg, d
                            )
                        else:
                            error_edge_1d_cond += estimates.get_scaled_local_errors(
                                mg, d
                            )
                    elif mg.dim == 0:
                        error_edge_0d += estimates.get_scaled_local_errors(mg, d)

                # Populate dictionary with proper fields
                out[solver_name][resol]["error_node_2d"] = error_node_2d
                out[solver_name][resol]["error_node_1d_cond"] = error_node_1d_cond
                out[solver_name][resol]["error_node_1d_bloc"] = error_node_1d_bloc
                out[solver_name][resol]["error_edge_1d_cond"] = error_edge_1d_cond
                out[solver_name][resol]["error_edge_1d_bloc"] = error_edge_1d_bloc
                out[solver_name][resol]["error_edge_0d"] = error_edge_0d
                out[solver_name][resol]["scaled_majorant"] = scaled_majorant

                # Print info in the console
                print("SUMMARY OF ERRORS")
                print("Error node 2D:", error_node_2d)
                print("Error node 1D [Conductive]:", error_node_1d_cond)
                print("Error node 1D [Blocking]:", error_node_1d_bloc)
                print("Error edge 1D [Conductive]:", error_edge_1d_cond)
                print("Error edge 1D [Blocking]:", error_edge_1d_bloc)
                print("Error edge 0D:", error_edge_0d)
                print("Scaled majorant:", scaled_majorant)
                print("\n")

                # Export to PARAVIEW. Change discretization method if desired
                if solver_name == "mpfa":
                    if mesh_size == 0.05:
                        paraview = pp.Exporter(gb, "mpfa_coarse", folder_name="out")
                    elif mesh_size == 0.025:
                        paraview = pp.Exporter(gb, "mpfa_intermediate", folder_name="out")
                    elif mesh_size == 0.0125:
                        paraview = pp.Exporter(gb, "mpfa_fine", folder_name="out")

                    paraview.write_vtu(["pressure", "diffusive_error"])


#%% Export
# Permutations
rows = len(numerical_methods) * len(mesh_resolutions)

# Intialize lists
numerical_method_name = []
mesh_resolution_name = []
col_2d_node = []
col_1d_node_cond = []
col_1d_node_bloc = []
col_1d_edge_cond = []
col_1d_edge_bloc = []
col_0d_edge = []
col_scaled_majorant = []

# Populate lists
for i in itertools.product(numerical_methods, mesh_resolutions):
    numerical_method_name.append(i[0])
    mesh_resolution_name.append(i[1])
    col_2d_node.append(out[i[0]][i[1]]["error_node_2d"])
    col_1d_node_cond.append(out[i[0]][i[1]]["error_node_1d_cond"])
    col_1d_node_bloc.append(out[i[0]][i[1]]["error_node_1d_bloc"])
    col_1d_edge_cond.append(out[i[0]][i[1]]["error_edge_1d_cond"])
    col_1d_edge_bloc.append(out[i[0]][i[1]]["error_edge_1d_bloc"])
    col_0d_edge.append(out[i[0]][i[1]]["error_edge_0d"])
    col_scaled_majorant.append(out[i[0]][i[1]]["scaled_majorant"])

# Prepare for exporting
export = np.zeros(
    rows,
    dtype=[
        ("var1", "U6"),
        ("var2", "U6"),
        ("var3", float),
        ("var4", float),
        ("var5", float),
        ("var6", float),
        ("var7", float),
        ("var8", float),
        ("var9", float),
    ],
)

export["var1"] = numerical_method_name
export["var2"] = mesh_resolution_name
export["var3"] = col_2d_node
export["var4"] = col_1d_node_cond
export["var5"] = col_1d_node_bloc
export["var6"] = col_1d_edge_cond
export["var7"] = col_1d_edge_bloc
export["var8"] = col_0d_edge
export["var9"] = col_scaled_majorant

header = "num_method mesh_size eta_omega_2d eta_omega_1d_c eta_omega_1d_b "
header += "eta_gamma_1d_c eta_gamma_1d_b eta_gamma_0d scaled_majorant "

np.savetxt(
    "convergence_bench2d.txt",
    export,
    delimiter=",",
    fmt="%4s %8s %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e",
    header=header,
)
