import mdestimates as mde
import porepy as pp
import numpy as np
import quadpy as qp

import mdestimates.estimates_utils as utils

from analytical_2d import ExactSolution2D

from typing import List


class TrueErrors2D(ExactSolution2D):
    def __init__(self, gb: pp.GridBucket, estimates: mde.ErrorEstimate):
        super().__init__(gb)
        self.estimates = estimates

    def __repr__(self) -> str:
        return "True error object for 2D validation"

    # Reconstructed pressures
    def reconstructed_p2d_p1(self) -> np.ndarray:
        """Compute reconstructed pressure for the matrix"""

        cc = self.g2d.cell_centers

        # Get reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        rp_cc_2d = np.zeros([self.g2d.num_cells, 1])
        for idx in self.cell_idx:
            rp_cc_2d += (
                p[0] * cc[0].reshape(self.g2d.num_cells, 1)
                + p[1] * cc[1].reshape(self.g2d.num_cells, 1)
                + p[2]
            ) * idx.reshape(self.g2d.num_cells, 1)

        return rp_cc_2d.flatten()

    def reconstructed_p1d_p1(self) -> np.ndarray:
        """Compute reconstructed pressure for the fracture"""

        # Get hold of reconstructed pressure
        recon_p = self.d1d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Rotate grid and get cell centers
        g_rot = mde.RotatedGrid(self.g1d)
        cc = g_rot.cell_centers

        # Project reconstructed pressure onto the cell centers
        rp_cc_1d = p[0] * cc[0].reshape(self.g1d.num_cells, 1) + p[1]

        return rp_cc_1d.flatten()

    # Reconstructed pressure gradients
    def reconstructed_gradp2_p1(self) -> List[np.ndarray]:
        """Compute reconstructed pressure gradient for the matrix"""

        # Get reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpx_cc_2d = p[0]
        gradpy_cc_2d = p[1]
        gradpz_cc_2d = np.zeros([self.g2d.num_cells, 1])

        gradp_cc = [
            gradpx_cc_2d.flatten(),
            gradpy_cc_2d.flatten(),
            gradpz_cc_2d.flatten(),
        ]

        return gradp_cc

    def reconstructed_gradp1_p1(self) -> List[np.ndarray]:
        """Compute reconstructed pressure gradient for the fracture"""

        # Get reconstructed pressure
        recon_p = self.d1d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpx_cc_1d = p[0]
        gradpy_cc_1d = np.zeros([self.g1d.num_cells, 1])
        gradpz_cc_1d = np.zeros([self.g1d.num_cells, 1])

        gradp_cc = [
            gradpx_cc_1d.flatten(),
            gradpy_cc_1d.flatten(),
            gradpz_cc_1d.flatten(),
        ]

        return gradp_cc

    # Reconstructed velocities
    def reconstructed_u2d(self) -> List[np.ndarray]:
        """Compute reconstructed velocity for the matrix"""

        cc = self.g2d.cell_centers

        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_2d_x = np.zeros([self.g2d.num_cells, 1])
        ru_cc_2d_y = np.zeros([self.g2d.num_cells, 1])
        ru_cc_2d_z = np.zeros([self.g2d.num_cells, 1])

        for idx in self.cell_idx:
            ru_cc_2d_x += (
                u[0] * cc[0].reshape(self.g2d.num_cells, 1) + u[1]
            ) * idx.reshape(self.g2d.num_cells, 1)
            ru_cc_2d_y += (
                u[0] * cc[1].reshape(self.g2d.num_cells, 1) + u[2]
            ) * idx.reshape(self.g2d.num_cells, 1)

        ru_cc_2d = [
            ru_cc_2d_x.flatten(),
            ru_cc_2d_y.flatten(),
            ru_cc_2d_z.flatten(),
        ]

        return ru_cc_2d

    def reconstructed_u1d(self) -> np.ndarray:
        """Compute reconstructed velocity for the fracture"""

        # Rotate embedded grid and
        g_rot = mde.RotatedGrid(self.g1d)
        cc = g_rot.cell_centers

        recon_u = self.d1d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_1d_x = u[0].flatten() * cc[0] + u[1].flatten()

        ru_cc_1d = ru_cc_1d_x

        return ru_cc_1d

    # Residual errors
    def residual_error(self) -> float:
        """Global matrix residual error using local Poincare constants"""

        residual_error_2d = np.sum(self.residual_error_2d()) ** 0.5

        residual_error_1d = np.sum(self.residual_error_1d()) ** 0.5

        residual_error = residual_error_1d + residual_error_2d

        return residual_error

    def residual_error_2d(self) -> np.ndarray:
        """Local matrix residual error squared for local Poincare constants"""

        weight_2d = (self.g2d.cell_diameters() / np.pi) ** 2
        residual_norm = self.residual_norm_sq_2d()

        residual_error_2d = weight_2d * residual_norm

        return residual_error_2d

    def residual_error_1d(self) -> np.ndarray:
        """Local fracture residual error squared for local Poincare constant"""

        weight_1d = (self.g1d.cell_diameters() / np.pi) ** 2
        residual_norm = self.residual_norm_sq_1d()

        residual_error_1d = weight_1d * residual_norm

        return residual_error_1d

    def residual_norm_sq_2d(self) -> np.ndarray:
        """Compute square of residual errors for 2D (onnly the norm)"""

        # Retrieve reconstructed velocity
        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Divergence of the reconstructed flux
        div_u = 2 * u[0]

        # Integration method and retrieving elements
        int_method = qp.t2.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g2d, self.g2d)

        integral = np.zeros(self.g2d.num_cells)
        for (f, idx) in zip(self.f2d("fun"), self.cell_idx):
            # Declare integrand
            def integrand(x):
                return (f(x[0], x[1]) * np.ones_like(x[0]) - div_u) ** 2

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    def residual_norm_sq_1d(self) -> np.ndarray:
        """Compute square of residual errors for 1D (onnly the norm)"""

        # Retrieve reconstructed velocity
        recon_u = self.d1d[self.estimates.estimates_kw]["recon_u"].copy()

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Divergence of the reconstructed flux
        div_u = u[0]

        # Jump in mortar fluxes
        jump_in_mortars = (
            self.d1d[self.estimates.estimates_kw]["mortar_jump"].copy() / self.g1d.cell_volumes
        ).reshape(self.g1d.num_cells, 1)

        # Integration method and retrieving elements
        method = qp.c1.newton_cotes_closed(10)
        g_rot = mde.RotatedGrid(self.g1d)
        elements = utils.get_quadpy_elements(self.g1d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        def integrand(y):
            return (self.f1d("fun")(y) - div_u + jump_in_mortars) ** 2

        integral = method.integrate(integrand, elements)

        return integral

    # Exact pressure error
    def pressure_error(self) -> float:

        true_error = np.sqrt(
            self.pressure_error_squared_2d().sum()
            + self.pressure_error_squared_1d().sum()
            + self.pressure_error_squared_mortar().sum()
        )

        return true_error

    def pressure_error_squared_2d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coefficients
        if self.estimates.p_recon_method in ["keilegavlen", "cochez"]:
            p = self.d2d[self.estimates.estimates_kw]["recon_p"]
        else:
            p = self.d2d[self.estimates.estimates_kw]["postprocessed_p"]

        p = utils.poly2col(p)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g2d, self.g2d)

        # Compute the true error for each subregion
        integral = np.zeros(self.g2d.num_cells)

        # Compute integrals
        for (gradp, idx) in zip(self.gradp2d("fun"), self.cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x):
                # gradp exact in x and y
                gradp_exact_x = gradp[0](x[0], x[1])
                gradp_exact_y = gradp[1](x[0], x[1])

                # gradp reconstructed in x and y
                if self.estimates.p_recon_method in ["keilegavlen", "cochez"]:
                    gradp_recon_x = p[0] * np.ones_like(x[0])
                    gradp_recon_y = p[1] * np.ones_like(x[1])
                else:
                    gradp_recon_x = 2 * p[0] * x[0] + p[1] * x[1] + p[2] * np.ones_like(x[0])
                    gradp_recon_y = p[1] * x[0] + 2 * p[3] * x[1] + p[4] * np.ones_like(x[1])

                # integral in x and y
                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2

                return int_x + int_y

            integral += method.integrate(integrand, elements) * idx

        return integral

    def pressure_error_squared_1d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coefficients
        recon_p = self.d1d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.c1.newton_cotes_closed(10)
        g_rot = mde.RotatedGrid(self.g1d)
        elements = utils.get_quadpy_elements(self.g1d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        # Compute the true error
        def integrand(x):
            # Exact pressure gradient
            gradp_exact_rot = -self.gradp1d("fun")(x)  # due to rotation (hardcoded)

            # Reconstructed pressure gradient
            if self.estimates.p_recon_method in ["cochez", "keilegavlen"]:
                gradp_recon_x = p[0] * np.ones_like(x[0])
            else:
                gradp_recon_x = 2 * p[0] * x + p[1] * np.ones_like(x)

            # Compute integral
            int_x = (gradp_exact_rot - gradp_recon_x) ** 2

            return int_x

        integral = method.integrate(integrand, elements)

        return integral

    def pressure_error_squared_mortar(self) -> np.ndarray:

        # Instantiate diffusive flux object to access private methods
        dfe = mde.DiffusiveError(self.estimates)

        # Get hold of grids and dictionaries
        g_l, g_h = self.gb.nodes_of_edge(self.e)
        mg = self.mg

        # Obtain the number of sides of the mortar grid
        num_sides = mg.num_sides()
        if num_sides == 2:
            sides = [-1, 1]
        else:
            sides = [1]

        # Loop over the sides of the mortar grid
        true_error = np.zeros(mg.num_cells)

        for side in sides:

            # Get rotated grids and sorted elements
            high_grid, frac_faces = dfe._sorted_highdim_edge_grid(g_h, g_l, mg, side)
            mortar_grid, mortar_cells = dfe._sorted_side_grid(mg, g_l, side)
            low_grid, low_cells = dfe._sorted_low_grid(g_l)

            # Merge the three grids into one
            merged_grid = dfe._merge_grids(low_grid, mortar_grid, high_grid)

            # Note that the following mappings are local for each merged grid.
            # For example, to retrieve the global fracture faces indices, we
            # should write frac_faces[merged_high_ele], and to retrieve the
            # global mortar cells, we should write
            # mortar_cells[merged_mortar_ele]
            # Retrieve element mapping from sorted grids to merged grid
            merged_high_ele = dfe._get_grid_uniongrid_elements(merged_grid, high_grid)
            merged_mortar_ele = dfe._get_grid_uniongrid_elements(merged_grid, mortar_grid)
            merged_low_ele = dfe._get_grid_uniongrid_elements(merged_grid, low_grid)

            # Get projected pressure jump, normal permeabilities, and
            # normal velocities
            pressure_jump, k_perp, _ = dfe._project_poly_to_merged_grid(
                self.e,
                self.de,
                [low_cells, mortar_cells, frac_faces],
                [merged_low_ele, merged_mortar_ele, merged_high_ele],
            )

            # Define integration method and obtain quadpy elements
            method = qp.c1.newton_cotes_closed(10)
            qp_ele = utils.get_qp_elements_from_union_grid_1d(merged_grid)
            qp_ele *= -1  # To use real coordinates

            # Define integrand
            def integrand(x):
                coors = x[np.newaxis, :, :]  # this is needed for 1D grids
                p_jump = utils.eval_p1(pressure_jump, -coors)  # -coors because is rotated
                return (self.p1d("fun")(x) - p_jump) ** 2

            # Evaluate integral
            diffusive_error_merged = method.integrate(integrand, qp_ele)

            # Sum errors corresponding to a mortar cell
            diffusive_error_side = np.zeros(len(mortar_cells))
            for mortar_element in range(len(mortar_cells)):
                idx = mortar_cells[mortar_element] == mortar_cells[merged_mortar_ele]
                diffusive_error_side[mortar_element] = diffusive_error_merged[idx].sum()

            # Append into the list
            true_error[mortar_cells] = diffusive_error_side

        return true_error