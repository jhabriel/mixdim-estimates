from __future__ import annotations
import porepy as pp
import numpy as np
import scipy.sparse as sps
import mdestimates as mde
import mdestimates.estimates_utils as utils


class PressureReconstruction(mde.ErrorEstimate):
    """Class for pressure reconstruction techniques"""

    def __init__(self, estimate: mde.ErrorEstimate):
        super().__init__(
            gb=estimate.gb,
            conservation=estimate.conservation,
            kw=estimate.kw,
            sd_operator_name=estimate.sd_operator_name,
            p_name=estimate.p_name,
            flux_name=estimate.flux_name,
            lam_name=estimate.lam_name,
            estimates_kw=estimate.estimates_kw,
            p_recon_method=estimate.p_recon_method,
            source_list=estimate.source_list,
            poincare_list=estimate.poincare_list
        )

    def reconstruct_pressure(self):
        """
        Reconstructs the pressure in all subdomains of the grid bucket.

        Returns
        -------
        None.

        Notes
        -----
        (*) The data dictionary of each node of the grid bucket is updated
        with the field d[self.estimates_kw]["recon_p"], a NumPy nd-array
        containing the coefficients of the reconstructed pressure.
        """

        # Loop through all subdomains
        for g, d in self.gb:

            # Handle the case of zero-dimensional subdomains
            if g.dim == 0:
                pcc = d[pp.STATE][self.p_name].copy()
                d[self.estimates_kw]["node_pressure"] = pcc
                d[self.estimates_kw]["recon_p"] = pcc
                continue

            # Rotate grid
            g_rot: mde.RotatedGrid = mde.RotatedGrid(g)

            # Obtain Lagrangian coordinates
            if self.p_recon_method == "keilegavlen":
                point_val, point_coo = self.keilegavlen_p1(g, g_rot, d)
            elif self.p_recon_method == "cochez-dhondt":
                point_val, point_coo = self.cochez_p1(g, g_rot, d)
            elif self.p_recon_method == "vohralik":
                point_val, point_coo = self.vohralik_p2(g, g_rot, d)
            else:
                raise ValueError("Pressure reconstruction method not "
                                 "available.")

            # Obtain pressure coefficients
            recons_p = utils.interpolate_P1(point_val, point_coo)

            # TEST: Pressure reconstruction
            self._test_pressure_reconstruction(g, recons_p, point_val, point_coo)
            # END TEST

            # Save in the data dictionary.
            d[self.estimates_kw]["recon_p"] = recons_p

        return None

    def cochez_p1(self, g: pp.Grid, g_rot: mde.RotatedGrid, d: dict):
        """
        Pressure reconstruction using average of P0 potentials over patches.

        Parameters
        ----------
        g : PorePy object
            Grid
        g_rot: Rotated grid object
            Rotated pseudo-grid
        d : Dictionary
            Dicitionary containing the parameters

        Raises
        ------
        Value Error:
            If pressure solution is not in d[pp.STATE]
            If pressure solution does not have the correct size

        Returns
        -------
        [point_val, point_coo]: Tuple
            List containing the pressures and coordinates at the Lagrangian points

        """

        # Sanity checks
        if self.p_name not in d[pp.STATE]:
            raise ValueError("Pressure solution not found.")

        if d[pp.STATE][self.p_name].size != g.num_cells:
            raise ValueError("Inconsistent size of pressure solution.")

        # Retrieve P0 cell-center pressure
        p_cc = d[pp.STATE][self.p_name].copy()

        # Retrieving topological data
        nc = g.num_cells
        nf = g.num_faces
        nn = g.num_nodes

        # Perform reconstruction
        cell_nodes = g.cell_nodes()
        cell_node_volumes = cell_nodes * sps.dia_matrix((g.cell_volumes, 0), (nc, nc))
        sum_cell_nodes = cell_node_volumes * np.ones(nc)
        cell_nodes_scaled = (
                sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes
        )

        cell_nodes_pressure = (
            sps.dia_matrix((1.0 / sum))
        )


        # Obtaining nodal pressures
        cell_nodes_map, _, _ = sps.find(g.cell_nodes())
        cell_node_matrix = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
        nodal_pressures = np.zeros(nn)

        for col in range(g.dim + 1):
            nodes = cell_node_matrix[:, col]
            dist = g_rot.nodes[: g.dim, nodes] - g_rot.cell_centers[: g.dim]
            scaling = cell_nodes_scaled[nodes, np.arange(nc)]
            contribution = (
                    np.asarray(scaling) * (p_cc + np.sum(dist * loc_grad, axis=0))
            ).ravel()
            nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)

        # Treatment of boundary conditions
        # TODO: Check this: We should only look for Dirichlet bc at the ambient grid
        bc = d[pp.PARAMETERS][self.kw]["bc"]
        bc_values = d[pp.PARAMETERS][self.kw]["bc_values"]
        external_dirichlet_boundary = np.logical_and(
            bc.is_dir, g.tags["domain_boundary_faces"]
        )
        face_vec = np.zeros(nf)
        face_vec[external_dirichlet_boundary] = 1
        num_dir_face_of_node = g.face_nodes * face_vec
        is_dir_node = num_dir_face_of_node > 0
        face_vec *= 0
        face_vec[external_dirichlet_boundary] = bc_values[external_dirichlet_boundary]
        node_val_dir = g.face_nodes * face_vec
        node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
        nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

        # Save in the dictionary
        d[self.estimates_kw]["node_pressure"] = nodal_pressures

        # Export lagrangian nodes and coordinates
        cell_nodes_map, _, _ = sps.find(g.cell_nodes())
        nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
        point_val = nodal_pressures[nodes_cell]
        point_coo = g_rot.nodes[:, nodes_cell]

        return point_val, point_coo

    def keilegavlen_p1(self, g: pp.Grid, g_rot: mde.RotatedGrid, d: dict):
        """
        Pressure reconstruction using the inverse of the numerical fluxes.
        Author: Eirik Keilegavlen

        Parameters
        ----------
        g : PorePy object
            Grid
        g_rot: Rotated grid object
            Rotated pseudo-grid
        d : Dictionary
            Dicitionary containing the parameters

        Raises
        ------
        Value Error:
            If pressure solution is not in d[pp.STATE]
            If pressure solution does not have the correct size
            If full fluxes are not in d[self.estimates_kw]

        Returns
        -------
        [point_val, point_coo]: Tuple
            List containing the pressures and coordinates at the Lagrangian points

        """

        # CHECK: Pressure solution in dictionary?
        if self.p_name not in d[pp.STATE]:
            raise ValueError("Pressure solution not found.")

        # CHECK: Pressure solution shape?
        if d[pp.STATE][self.p_name].size != g.num_cells:
            raise ValueError("Inconsistent size of pressure solution.")

        # CHECK: Full flux in dictionary?
        if "full_flux" not in d[self.estimates_kw]:
            raise ValueError("Full flux must be computed first")

        # Retrieve P0 cell-center pressure
        p_cc = d[pp.STATE][self.p_name].copy()

        # Retrieve full fluxes
        flux = d[self.estimates_kw]["full_flux"].copy()

        # Retrieving topological data
        nc = g.num_cells
        nf = g.num_faces
        nn = g.num_nodes

        # Perform reconstruction
        cell_nodes = g.cell_nodes()
        cell_node_volumes = cell_nodes * sps.dia_matrix((g.cell_volumes, 0), (nc, nc))
        sum_cell_nodes = cell_node_volumes * np.ones(nc)
        cell_nodes_scaled = (
                sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes
        )

        # Project fluxes using RT0
        # d_RT0 = d.copy()
        # pp.RT0(self.kw).discretize(g, d_RT0)
        # proj_flux = pp.RT0(self.kw).project_flux(g, flux, d_RT0)[: g.dim]

        # Retrieve reconstructed velocities
        coeff = d[self.estimates_kw]["recon_u"]
        if g.dim == 3:
            proj_flux = np.array(
                [
                    coeff[:, 0] * g_rot.cell_centers[0] + coeff[:, 1],
                    coeff[:, 0] * g_rot.cell_centers[1] + coeff[:, 2],
                    coeff[:, 0] * g_rot.cell_centers[2] + coeff[:, 3],
                ]
            )
        elif g.dim == 2:
            proj_flux = np.array(
                [
                    coeff[:, 0] * g_rot.cell_centers[0] + coeff[:, 1],
                    coeff[:, 0] * g_rot.cell_centers[1] + coeff[:, 2],
                ]
            )
        else:
            proj_flux = np.array(
                [
                    coeff[:, 0] * g_rot.cell_centers[0] + coeff[:, 1],
                ]
            )

        # Obtain local gradients
        loc_grad = np.zeros((g.dim, nc))
        perm = d[pp.PARAMETERS][self.kw]["second_order_tensor"].values
        for ci in range(nc):
            loc_grad[: g.dim, ci] = -np.linalg.inv(perm[: g.dim, : g.dim, ci]).dot(
                proj_flux[:, ci]
            )

        # Obtaining nodal pressures
        cell_nodes_map, _, _ = sps.find(g.cell_nodes())
        cell_node_matrix = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
        nodal_pressures = np.zeros(nn)

        for col in range(g.dim + 1):
            nodes = cell_node_matrix[:, col]
            dist = g_rot.nodes[: g.dim, nodes] - g_rot.cell_centers[: g.dim]
            scaling = cell_nodes_scaled[nodes, np.arange(nc)]
            contribution = (
                    np.asarray(scaling) * (p_cc + np.sum(dist * loc_grad, axis=0))
            ).ravel()
            nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)

        # Treatment of boundary conditions
        # TODO: Check this: We should only look for Dirichlet bc at the ambient grid
        bc = d[pp.PARAMETERS][self.kw]["bc"]
        bc_values = d[pp.PARAMETERS][self.kw]["bc_values"]
        external_dirichlet_boundary = np.logical_and(
            bc.is_dir, g.tags["domain_boundary_faces"]
        )
        face_vec = np.zeros(nf)
        face_vec[external_dirichlet_boundary] = 1
        num_dir_face_of_node = g.face_nodes * face_vec
        is_dir_node = num_dir_face_of_node > 0
        face_vec *= 0
        face_vec[external_dirichlet_boundary] = bc_values[external_dirichlet_boundary]
        node_val_dir = g.face_nodes * face_vec
        node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
        nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

        # Save in the dictionary
        d[self.estimates_kw]["node_pressure"] = nodal_pressures

        # Export lagrangian nodes and coordintates
        cell_nodes_map, _, _ = sps.find(g.cell_nodes())
        nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
        point_val = nodal_pressures[nodes_cell]
        point_coo = g_rot.nodes[:, nodes_cell]

        return point_val, point_coo

    def vohralik_p2(self, g: pp.Grid, g_rot: mde.RotatedGrid, d: dict):
        pass

    def _test_pressure_reconstruction(self, g, recon_p, point_val, point_coo):
        """
        Testing pressure reconstruction. This function uses the reconstructed
        pressure local polynomial and perform an evaluation at the Lagrangian
        points, and checks if the those values are equal to the point_val array.

        Parameters
        ----------
        g : PorePy object
            Grid.
        recon_p : NumPy nd-Array
            Reconstructed pressure polynomial.
        point_val : NumPy nd-Array
            Pressure avlues at the Lagrangian nodes.
        point_coo : NumPy array
            Coordinates at the Lagrangian nodes.

        Returns
        -------
        None.

        """

        def assert_reconp(eval_poly, point_val):
            np.testing.assert_allclose(
                eval_poly,
                point_val,
                rtol=1e-6,
                atol=1e-3,
                err_msg="Pressure reconstruction has failed",
            )

        eval_poly = utils.eval_P1(recon_p, point_coo)
        assert_reconp(eval_poly, point_val)

        return None
