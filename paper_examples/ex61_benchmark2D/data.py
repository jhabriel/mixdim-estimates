import numpy as np
import porepy as pp

# Disclaimer: Script copied or partially modified from 10.5281/zenodo.3374624


def add_data(gb, flow_dir, tol):

    for g, d in gb:
        d["Aavatsmark_transmissibilities"] = True
        d["is_tangential"] = True

        is_fv = issubclass(type(d["discr"]["scheme"]), pp.FVElliptic)

        # Assign apertures
        aperture = np.power(1e-4, 2 - g.dim) * np.ones(g.num_cells)

        # set the permeability
        if g.dim == 2:
            kxx = aperture
            if is_fv:
                perm = pp.SecondOrderTensor(kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        elif g.dim == 1:
            if d["is_low"]:
                kxx = 1e-4 * aperture
            else:
                kxx = 1e4 * aperture
            if is_fv:
                perm = pp.SecondOrderTensor(kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(kxx=kxx, kyy=1, kzz=1)
        elif g.dim == 0:
            perm = None

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        if b_faces.size != 0:

            b_face_centers = g.face_centers[:, b_faces]

            if flow_dir == "top_to_bottom":
                b_dir_low = b_face_centers[1, :] < 0 + tol
                b_dir_high = b_face_centers[1, :] > 1 - tol
            else:
                b_dir_low = b_face_centers[0, :] > 1 - tol
                b_dir_high = b_face_centers[0, :] < 0 + tol

            labels = np.array(["neu"] * b_faces.size)
            labels[np.logical_or(b_dir_low, b_dir_high)] = "dir"
            bc = pp.BoundaryCondition(g, b_faces, labels)

            bc_val[b_faces[b_dir_low]] = 1
            bc_val[b_faces[b_dir_high]] = 4

        else:
            bc = pp.BoundaryCondition(g, np.empty(0), np.empty(0))

        specified_parameters = {
            "second_order_tensor": perm,
            "bc": bc,
            "bc_values": bc_val,
        }
        pp.initialize_default_data(g, d, "flow", specified_parameters)

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    for e, d in gb.edges():
        gl, gh = gb.nodes_of_edge(e)
        if gl.dim == 1:
            if gb.node_props(gl, "is_low"):
                kxx = 1e-4
            else:
                kxx = 1e4
        else:  # gl.dim == 0
            if d["is_low"]:
                kxx = 2 / (1 / 1e-4 + 1 / 1e4)
            else:
                kxx = 1e4

        mg = d["mortar_grid"]

        aperture_h = np.power(1e-4, 2 - gh.dim)
        kn = 2 * kxx * np.ones(mg.num_cells) / 1e-4 * aperture_h

        d[pp.PARAMETERS] = pp.Parameters(mg, "flow", {"normal_diffusivity": kn})
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
