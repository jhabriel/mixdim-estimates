import mdestimates
import porepy as pp
import numpy as np
import scipy.sparse as sps
import mdestimates.estimates_utils as utils


class VelocityReconstruction(mdestimates.ErrorEstimate):
    """ Main class for RT0 velocity reconstruction."""

    def __init__(self, gb: pp.GridBucket):
        super().__init__(gb)

    def compute_full_flux(self):
        """
        Computes the full flux in all subdomains of the grid bucket.

        Technical notes:
        ----------------
        (*) The full flux is composed by the subdomain Darcy flux plus the projection of the
            lower-dimensional mortar fluxes.
        (*) The data dictionary of each subdomain is updated with the field
            d[self.estimates_kw]["full_flux"], a NumPy array of length g.num_faces.
        (*) Full fluxes are not defined for 0d-subdomains. In this case we
            assign a None variable to the field d[self.estimates_kw]["full_flux"].
        """

        # Loop through all the nodes of the grid bucket
        for g, d in self.gb:

            # Handle the case of zero-dimensional subdomains
            if g.dim == 0:
                d[self.estimates_kw]["full_flux"] = None
                continue

            # Retrieve subdomain discretization
            discr = d[pp.DISCRETIZATION][self.p_name][self.sd_operator_name]

            # Boolean variable for checking is the scheme is FV
            is_fv = issubclass(type(discr), pp.FVElliptic)

            if is_fv:  # fvm-schemes

                # Compute Darcy flux
                parameter_dictionary = d[pp.PARAMETERS][self.kw]
                matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][self.kw]
                darcy_flux = (
                        matrix_dictionary["flux"] * d[pp.STATE][self.p_name].copy()
                        + matrix_dictionary["bound_flux"] * parameter_dictionary["bc_values"]
                )

                # Add tcontribution of mortar fluxes from edges associated
                # to the high-dim subdomain
                induced_flux = np.zeros(darcy_flux.size)
                faces = g.tags["fracture_faces"]
                if np.any(faces):
                    for _, d_e in self.gb.edges_of_node(g):
                        g_m = d_e["mortar_grid"]
                        if g_m.dim == g.dim:
                            continue
                        # project the mortar variable back to the high-dim subdomain
                        induced_flux += (
                                matrix_dictionary["bound_flux"]
                                * g_m.mortar_to_primary_int()
                                * d_e[pp.STATE][self.lam_name].copy()
                        )

            else:  # fem-schemes

                # Retrieve Darcy flux from the solution array
                if self.flux_name not in d[pp.STATE]:
                    raise ValueError("FEM flux solution array must be explicitly passed.")
                else:
                    darcy_flux = d[pp.STATE][self.flux_name]

                # We need to recover the flux from the mortar variable before
                # the projection, only lower dimensional edges need to be considered.
                induced_flux = np.zeros(darcy_flux.size)
                faces = g.tags["fracture_faces"]
                if np.any(faces):
                    # recover the sign of the flux, since the mortar is assumed
                    # to point from the higher to the lower dimensional problem
                    _, indices = np.unique(g.cell_faces.indices, return_index=True)
                    sign = sps.diags(g.cell_faces.data[indices], 0)

                    for _, d_e in self.gb.edges_of_node(g):
                        g_m = d_e["mortar_grid"]
                        if g_m.dim == g.dim:
                            continue
                        # project the mortar variable back to the high-dim subdomain
                        induced_flux += (
                                sign
                                * g_m.primary_to_mortar_avg().T
                                * d_e[pp.STATE][self.lam_name].copy()
                        )

            # Store in the data dictionary
            d[self.estimates_kw]["full_flux"] = darcy_flux + induced_flux

        return None

    def reconstruct_velocity(self):
        """
        Computes flux reconstruction using RT0 extension of normal full fluxes.

        Technical Note:
        ---------------
        The data dictionary of each node of the grid bucket will be updated
        with the field d[self.estimates_kw]["recons_vel"], a NumPy nd-array
        of shape (g.num_cells x (g.dim+1)) containing the coefficients of the
        reconstructed velocity for each element. Each column corresponds to the
        coefficient a, b, c, and so on.

        The coefficients satisfy the following velocity fields depending on the
        dimensionality of the problem:

        q = ax + b                          (for 1d),
        q = (ax + b, ay + c)^T              (for 2d),
        q = (ax + b, ay + c, az + d)^T      (for 3d).

        The reconstructed velocity field inside an element K is given by:

            q = sum_{j=1}^{g.dim+1} q_j psi_j,

        where psi_j are the global basis functions defined on each face,
        and q_j are the normal fluxes.

        The global basis takes the form

        psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i)^T                     (for 1d),
        psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i)^T            (for 2d),
        psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i, z - z_i)^T   (for 3d),

        where s(normal_j) is the sign of the normal vector,|K| is the Lebesgue
        measure of the element K, and (x_i, y_i, z_i) are the coordinates of the
        opposite side nodes to the face j. The function s(normal_j) = 1 if the
        signs of the local and global normals are the same, and -1 otherwise.
        """

        # Loop through all the nodes of the grid bucket
        for g, d in self.gb:

            # Handle the case of zero-dimensional subdomains
            if g.dim == 0:
                d[self.estimates_kw]["recon_u"] = None
                continue

            # First, rotate the grid. Note that if g == gb.dim_max(), this has no effect.
            g_rot = utils.rotate_embedded_grid(g)

            # Useful mappings
            cell_faces_map, _, _ = sps.find(g.cell_faces)
            cell_nodes_map, _, _ = sps.find(g.cell_nodes())

            # Cell-basis arrays
            faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))
            opp_nodes_cell = utils.get_opposite_side_nodes(g)
            opp_nodes_coor_cell = g_rot.nodes[:, opp_nodes_cell]
            sign_normals_cell = utils.get_sign_normals(g, g_rot)
            vol_cell = g.cell_volumes

            # Retrieve full flux from the data dictionary
            full_flux = d[self.estimates_kw]["full_flux"]

            # TEST -> Local mass conservation
            # Check if mass conservation is satisfied on a cell basis, in order to do
            # this, we check on a local basis, if the divergence of the flux equals
            # the sum of internal and external source terms
            full_flux_local_div = (sign_normals_cell * full_flux[faces_cell]).sum(axis=1)
            external_src = d[pp.PARAMETERS][self.kw]["source"]
            internal_src = self._internal_source_term_contribution(g)
            np.testing.assert_allclose(
                full_flux_local_div,
                external_src + internal_src,
                rtol=1e-6,
                atol=1e-3,
                err_msg="Error estimates only valid for local mass-conservative methods.",
            )
            # END OF TEST

            # Perform actual reconstruction and obtain coefficients
            coeffs = np.empty([g.num_cells, g.dim + 1])
            alpha = 1 / (g.dim * vol_cell)
            coeffs[:, 0] = alpha * np.sum(sign_normals_cell * full_flux[faces_cell], axis=1)
            for dim in range(g.dim):
                coeffs[:, dim + 1] = -alpha * np.sum(
                    (sign_normals_cell * full_flux[faces_cell] * opp_nodes_coor_cell[dim]),
                    axis=1,
                )

            # TEST -> Flux reconstruction
            # Check if the reconstructed evaluated at the face centers normal fluxes
            # match the numerical ones
            recons_flux = self._reconstructed_face_fluxes(g, g_rot, coeffs)
            np.testing.assert_almost_equal(
                recons_flux,
                full_flux,
                decimal=12,
                err_msg="Flux reconstruction has failed.",
            )
            # END OF TEST

            # Store coefficients in the data dictionary
            d[self.estimates_kw]["recon_u"] = coeffs

        return None

    def _internal_source_term_contribution(self, g):
        """
        Obtain flux contribution from higher-dimensional neighboring interfaces
        to lower-dimensional subdomains in the form of internal source terms

        Parameters
        ----------
        g : PorePy object
            Grid.

        Returns
        -------
        internal_source : NumPy array (g.num_cells)
            Flux contribution from higher-dimensional neighboring interfaces to the
            lower-dimensional grid g, in the form of a source term.

        """

        # Initialize internal source term
        internal_source = np.zeros(g.num_cells)

        # Handle the case of mono-dimensional grids
        if self.gb.num_graph_nodes() == 1:
            return internal_source

        # Obtain higher dimensional neighboring nodes
        g_highs = self.gb.node_neighbors(g, only_higher=True)

        # We loop through all the high-dim adjacent interfaces to the low-dim
        # subdomain to map the mortar fluxes to internal source terms
        for g_high in g_highs:
            # Retrieve the dictionary and mortar grid of the corresponding edge
            d_edge = self.gb.edge_props((g, g_high))
            g_mortar = d_edge["mortar_grid"]

            # Retrieve mortar fluxes
            mortar_flux = d_edge[pp.STATE][self.lam_name]

            # Obtain source term contribution associated to the neighboring interface
            internal_source += g_mortar.mortar_to_secondary_int() * mortar_flux

        return internal_source

    @staticmethod
    def _reconstructed_face_fluxes(g: pp.Grid, g_rot, coeff):
        """
        Obtain reconstructed fluxes at the cell centers for a given mesh

        Parameters
        ----------
        g : PorePy object
            Grid
        g_rot: Rotated object
            Rotated pseudo-grid
        coeff : NumPy array of shape (g.num_cells x (g.dim+1))
            Coefficients of the reconstructed velocity field

        Returns
        -------
        recons_face_fluxes : NumPy array of shape g.num_faces
           Reconstructed face-centered fluxes

        """

        # Mappings
        cell_faces_map, _, _ = sps.find(g.cell_faces)
        faces_cell = cell_faces_map.reshape(np.array([g.num_cells, g.dim + 1]))

        # Normal and face center coordinates of each cell
        normal_faces_cell = g_rot.face_normals[:, faces_cell]
        facenters_cell = g_rot.face_centers[:, faces_cell]

        # Reconstructed velocity at each face-center of each cell
        # TODO: Avoid the double loop
        q_rec = np.zeros([g.dim, g.num_cells, g.dim + 1])
        for dim in range(g.dim):
            for cell in range(g.num_cells):
                q_rec[dim][cell] = np.array(
                    [coeff[cell, 0] * facenters_cell[dim][cell] + coeff[cell, dim + 1]]
                )

        # Reconstructed flux at each face-center of each cell
        # TODO: Avoid the double loop
        Q_rec = np.zeros([g.num_cells, g.dim + 1])
        for dim in range(g.dim):
            for cell in range(g.num_cells):
                Q_rec[cell] += q_rec[dim][cell] * normal_faces_cell[dim][cell]
        Q_flat = Q_rec.flatten()
        idx_q = np.array(
            [np.where(faces_cell.flatten() == x)[0][0] for x in range(g.num_faces)]
        )
        recons_face_fluxes = Q_flat[idx_q]

        return recons_face_fluxes