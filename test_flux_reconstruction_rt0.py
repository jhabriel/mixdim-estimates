# -*- coding: utf-8 -*-
import numpy as np
import porepy as pp
import scipy.sparse as sps
import unittest

from itertools import product
from ReconstructFluxRT0 import *


def _computeReconFaceFluxes(g, coeff):
    # Mappings
    cell_faces_map, _, _ = sps.find(g.cell_faces)
    faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

    # Normal and face center coordinates of each cell
    normal_faces_cell = np.empty([g.dim, g.num_cells, g.dim + 1])
    facenters_cell = np.empty([g.dim, g.num_cells, g.dim + 1])
    for dim in range(g.dim):
        normal_faces_cell[dim] = g.face_normals[dim][faces_cell]
        facenters_cell[dim] = g.face_centers[dim][faces_cell]

    # Reconstructed velocity at each face-center of each cell
    q_rec = np.empty([g.dim, g.num_cells, g.dim + 1])
    for dim, cell in product(range(g.dim), range(g.num_cells)):
        q_rec[dim][cell] = np.array(
            [coeff[cell, 0] * facenters_cell[dim][cell] + coeff[cell, dim + 1]]
        )

    # Reconstructed flux at each face-center of each cell
    Q_rec = np.empty([g.num_cells, g.dim + 1])
    for dim, cell in product(range(g.dim), range(g.num_cells)):
        Q_rec[cell] += q_rec[dim][cell] * normal_faces_cell[dim][cell]
    Q_flat = Q_rec.flatten()
    idx_q = np.array(
        [np.where(faces_cell.flatten() == x)[0][0] for x in range(g.num_faces)]
    )
    out = Q_flat[idx_q]

    return out


def perturb(g, rate, dx):

    rand = np.vstack((np.random.rand(g.dim, g.num_nodes), np.repeat(0.0, g.num_nodes)))
    g.nodes += rate * dx * (rand - 0.5)
    # Ensure there are no perturbations in the z-coordinate
    if g.dim == 2:
        g.nodes[2, :] = 0

    return g


def dist(n):
    return np.dot(pp.map_geometry.rotation_matrix(np.linalg.norm(n), [0, 0, 1]), n)


def make_gmsh_grid(mesh_size=0.1, L=[1, 1]):
    """
    Create an unstructured triangular mesh using Gmsh.

    Parameters:
        mesh_size (scalar): (approximated) size of triangular elements [-]
        L (array): length of the domain for each dimension [m]

    Returns:
        gb (PorePy object): PorePy grid bucket object containing all the grids
                            In this case we only have one grid.
    """

    domain = {"xmin": 0.0, "xmax": L[0], "ymin": 0.0, "ymax": L[1]}
    network_2d = pp.FractureNetwork2d(None, None, domain)
    target_h_bound = target_h_fracture = target_h_min = mesh_size

    mesh_args = {
        "mesh_size_bound": target_h_bound,
        "mesh_size_frac": target_h_fracture,
        "mesh_size_min": target_h_min,
    }

    gb = network_2d.mesh(mesh_args)

    return gb


class TestFluxReconstruction(unittest.TestCase):
    def test_zero_fluxes_one_cell_2d(self):
        """
        Test if the the reconstruction works for the smallest triangular
        structured domain while imposing zero flux everywhere
        """
        g = pp.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()
        kw_f = "flow"
        d = pp.initialize_default_data(g, {}, kw_f)
        d[pp.PARAMETERS][kw_f]["darcy_flux"] = np.zeros(g.num_faces)
        fv_flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]
        coeff, _ = FluxReconstructionRT0(g, d, kw_f).reconstruct_flux()
        rec_flux = _computeReconFaceFluxes(g, coeff)
        self.assertTrue(np.allclose(fv_flux, rec_flux))

    def test_random_fluxes_structured_2d(self):
        """
        Same case as before, but this time we use random fluxes.
        We check 5 randomly generated sizes of grids.
        """
        perm = np.random.permutation(5) + 1
        for perms in perm:
            g = pp.StructuredTriangleGrid([perms, perms], [1, 1])
            g.compute_geometry()
            kw_f = "flow"
            d = pp.initialize_default_data(g, {}, kw_f)
            d[pp.PARAMETERS][kw_f]["darcy_flux"] = np.random.rand(g.num_faces)
            fv_flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]
            coeff, _ = FluxReconstructionRT0(g, d, kw_f).reconstruct_flux()
            rec_flux = _computeReconFaceFluxes(g, coeff)
            self.assertTrue(np.allclose(fv_flux, rec_flux))

    def test_distorted_mesh_2d(self):
        """
        For random fluxes at the faces, randomly perturb the mesh
        """
        perturbations = 5
        for p in range(perturbations):
            n = 10
            g = pp.StructuredTriangleGrid([n, n / 2], [1.2, 5.33])
            g = perturb(g, 0.1, 1 / n)
            g.compute_geometry()
            kw_f = "flow"
            d = pp.initialize_default_data(g, {}, kw_f)
            d[pp.PARAMETERS][kw_f]["darcy_flux"] = np.random.rand(g.num_faces)
            fv_flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]
            coeff, _ = FluxReconstructionRT0(g, d, kw_f).reconstruct_flux()
            rec_flux = _computeReconFaceFluxes(g, coeff)
            self.assertTrue(np.allclose(fv_flux, rec_flux))

    def test_linear_pressure_drop_2d(self):
        """
        Having a linear pressure drop as a result, check if the reconstructed
        fluxes at the faces match the numerical solution
        """
        n = [3, 3]
        g = pp.StructuredTriangleGrid(n, [1, 1])
        g.compute_geometry()
        kw_f = "flow"
        subdomain_variable = "pressure"

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]

        labels = np.array(["dir"] * b_faces.size)
        labels[g.face_centers[1, b_faces] == 0] = "neu"
        labels[g.face_centers[1, b_faces] == 1.0] = "neu"

        bc = pp.BoundaryCondition(g, b_faces, labels)

        bc_val = np.zeros(g.num_faces)
        left = b_faces[g.face_centers[0, b_faces] == 1.0]
        bc_val[left] = np.ones(left.size)

        specified_parameters = {"bc": bc, "bc_values": bc_val}
        d = pp.initialize_default_data(g, {}, kw_f, specified_parameters)

        solver = pp.Tpfa(kw_f)
        solver.discretize(g, d)

        A, b = solver.assemble_matrix_rhs(g, d)
        p = sps.linalg.spsolve(A, b)
        _ = pp.set_state(d, {subdomain_variable: p})

        pp.fvutils.compute_darcy_flux(g, data=d)
        fv_flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]

        coeff, _ = FluxReconstructionRT0(g, d, kw_f).reconstruct_flux()
        rec_flux = _computeReconFaceFluxes(g, coeff)

        self.assertTrue(np.allclose(fv_flux, rec_flux))

    def test_rotated_grid_2d(self):
        """
        Test if the reconstruction is succesfull for a rotated grid
        with randomly generated fluxes at the faces
        """
        g = pp.StructuredTriangleGrid([10, 10], [2, 2])
        g.nodes[0, :] = g.nodes[0, :] - 1
        g.nodes[1, :] = g.nodes[1, :] - 1
        g.nodes = np.apply_along_axis(dist, 0, g.nodes)
        g.compute_geometry()

        perturbations = 5
        for p in range(perturbations):
            kw_f = "flow"
            d = pp.initialize_default_data(g, {}, kw_f)
            d[pp.PARAMETERS][kw_f]["darcy_flux"] = np.random.rand(g.num_faces)
            fv_flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]
            coeff, _ = FluxReconstructionRT0(g, d, kw_f).reconstruct_flux()
            rec_flux = _computeReconFaceFluxes(g, coeff)
            self.assertTrue(np.allclose(fv_flux, rec_flux))

    def test_gmsh_grid_2d(self):
        """
        Test if the reconstruction is succesfull for a unstructured grid
        generated using gmsh
        """
        gb = make_gmsh_grid()
        g = gb.grids_of_dimension(2)[0]

        perturbations = 5
        for p in range(perturbations):
            kw_f = "flow"
            d = pp.initialize_default_data(g, {}, kw_f)
            d[pp.PARAMETERS][kw_f]["darcy_flux"] = np.random.rand(g.num_faces)
            fv_flux = d[pp.PARAMETERS][kw_f]["darcy_flux"]
            coeff, _ = FluxReconstructionRT0(g, d, kw_f).reconstruct_flux()
            rec_flux = _computeReconFaceFluxes(g, coeff)
            self.assertTrue(np.allclose(fv_flux, rec_flux))


if __name__ == "__main__":
    unittest.main()
