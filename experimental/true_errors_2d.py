import mdestimates as mde
import porepy as pp
import numpy as np
import sympy as sym
import quadpy as qp

import mdestimates.estimates_utils as utils
from mdestimates._velocity_reconstruction import _internal_source_term_contribution as mortar_jump
from analytical_2d import ExactSolution2D
from typing import Tuple, List


class TrueErrors2D(ExactSolution2D):

    def __init__(self, gb: pp.GridBucket, estimates: mde.ErrorEstimate):
        super().__init__(gb)
        self.estimates = estimates

    def __repr__(self) -> str:
        return "True error object for 2D validation"

    # Reconstructed pressures
    def reconstructed_p2d(self) -> np.ndarray:

        cc = self.g2d.cell_centers

        # Get reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        rp_cc_2d = np.zeros([self.g2d.num_cells, 1])
        for idx in self.cell_idx:
            rp_cc_2d += (p[0] * cc[0].reshape(self.g2d.num_cells, 1)
                         + p[1] * cc[1].reshape(self.g2d.num_cells, 1)
                         + p[2]
                         ) * idx.reshape(self.g2d.num_cells, 1)

        return rp_cc_2d.flatten()

    def reconstructed_p1d(self) -> np.ndarray:

        # Get hold of reconstructed pressure
        recon_p = self.d1d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Rotate grid and get cell centers
        g_rot = utils.rotate_embedded_grid(self.g1d)
        cc = g_rot.cell_centers

        # Project reconstructed pressure onto the cell centers
        rp_cc_1d = p[0] * cc[0].reshape(self.g1d.num_cells, 1) + p[1]

        return rp_cc_1d.flatten()

    # Reconstructed pressure gradients
    def reconstructed_gradp2(self) -> List[np.ndarray]:

        # Get reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpx_cc_2d = p[0]
        gradpy_cc_2d = p[1]
        gradpz_cc_2d = np.zeros([self.g2d.num_cells, 1])

        gradp_cc = [gradpx_cc_2d.flatten(), gradpy_cc_2d.flatten(), gradpz_cc_2d.flatten()]

        return gradp_cc

    def reconstructed_gradp1(self) -> List[np.ndarray]:

        # Get reconstructed pressure
        recon_p = self.d1d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpx_cc_1d = p[0]
        gradpy_cc_1d = np.zeros([self.g1d.num_cells, 1])
        gradpz_cc_1d = np.zeros([self.g1d.num_cells, 1])

        gradp_cc = [gradpx_cc_1d.flatten(), gradpy_cc_1d.flatten(), gradpz_cc_1d.flatten()]

        return gradp_cc

    # Reconstructed velocities
    def reconstructed_u2d(self) -> List[np.ndarray]:

        cc = self.g2d.cell_centers

        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_2d_x = np.zeros([self.g2d.num_cells, 1])
        ru_cc_2d_y = np.zeros([self.g2d.num_cells, 1])
        ru_cc_2d_z = np.zeros([self.g2d.num_cells, 1])

        for idx in self.cell_idx:
            ru_cc_2d_x += (u[0] * cc[0].reshape(self.g2d.num_cells, 1) + u[1]) * idx.reshape(
                self.g2d.num_cells, 1)
            ru_cc_2d_y += (u[0] * cc[1].reshape(self.g2d.num_cells, 1) + u[2]) * idx.reshape(
                self.g2d.num_cells, 1)

        ru_cc_2d = [ru_cc_2d_x.flatten(), ru_cc_2d_y.flatten(), ru_cc_2d_z.flatten()]

        return ru_cc_2d

    def reconstructed_u1d(self) -> np.ndarray:

        # Rotate embedded grid and
        g_rot = utils.rotate_embedded_grid(self.g1d)
        cc = g_rot.cell_centers

        recon_u = self.d1d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_1d_x = u[0].flatten() * cc[0] + u[1].flatten()

        ru_cc_1d = ru_cc_1d_x

        return ru_cc_1d

    # Residual errors
    def residual_error_local_poincare(self) -> float:

        weight_2d = (self.g2d.cell_diameters()/np.pi) ** 2
        residual_error_2d = (weight_2d * self.residual_error_squared_2d()).sum() ** 0.5

        weight_1d = (self.g1d.cell_diameters()/np.pi) ** 2
        residual_error_1d = (weight_1d * self.residual_error_squared_1d()).sum() ** 0.5

        return residual_error_2d + residual_error_1d

    def residual_error_global_poincare(self) -> float:

        weight = (1 / (np.sqrt(2) * np.pi)) ** 2
        residual_error_2d = (weight * self.residual_error_squared_2d()).sum() ** 0.5
        residual_error_1d = (weight * self.residual_error_squared_1d()).sum() ** 0.5

        return residual_error_2d + residual_error_1d

    def residual_error_squared_2d(self) -> np.ndarray:

        # Retrieve reconstructed velocity
        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Diveregence of the reconstructed flux
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

        residual_error_squared = integral

        return residual_error_squared

    def residual_error_squared_1d(self) -> np.ndarray:

        # Retrieve reconstructed velocity
        recon_u = self.d1d[self.estimates.estimates_kw]["recon_u"].copy()

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Diveregence of the reconstructed flux
        div_u = u[0]

        # Jump in mortar fluxes
        jump_in_mortars = (mortar_jump(self.estimates, self.g1d) /
                           self.g1d.cell_volumes).reshape(
            self.g1d.num_cells, 1)

        # Integration method and retrieving elements
        method = qp.c1.newton_cotes_closed(10)
        g_rot = utils.rotate_embedded_grid(self.g1d)
        elements = utils.get_quadpy_elements(self.g1d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        def integrand(y):
            return (self.f1d("fun")(y) - div_u + jump_in_mortars) ** 2

        integral = method.integrate(integrand, elements)
        residual_error_squared = integral

        return residual_error_squared

    # Exact pressure error
    def pressure_error(self) -> float:

        true_error = (self.pressure_error_squared_2d().sum()
                      + self.pressure_error_squared_1d().sum()
                      + self.pressure_error_squared_mortar().sum()
                      ) ** 0.5

        return true_error

    def pressure_error_squared_2d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

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
                gradp_recon_x = pr[0] * np.ones_like(x[0])
                gradp_recon_y = pr[1] * np.ones_like(x[1])
                # integral in x and y
                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2
                return int_x + int_y
            integral += method.integrate(integrand, elements) * idx

        return integral

    def pressure_error_squared_1d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = self.d1d[self.estimates.estimates_kw]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.c1.newton_cotes_closed(10)
        g_rot = utils.rotate_embedded_grid(self.g1d)
        elements = utils.get_quadpy_elements(self.g1d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        print()
        # Compute the true error
        def integrand(x):
            # Exact pressure gradient
            gradp_exact_rot = -self.gradp1d("fun")(x)  # due to rotation (hardcoded)
            # Reconstructed pressure gradient
            gradp_recon_x = pr[0] * np.ones_like(x[0])
            # integral
            int_x = (gradp_exact_rot - gradp_recon_x) ** 2
            return int_x
        integral = method.integrate(integrand, elements)

        return integral

    def pressure_error_squared_mortar(self) -> np.ndarray:

        # Import functions
        from mdestimates._error_evaluation import (
            _sorted_highdim_edge_grid,
            _sorted_side_grid,
            _sorted_low_grid,
            _merge_grids,
            _get_grid_uniongrid_elements,
            _project_poly_to_merged_grid,
        )

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
            high_grid, frac_faces = _sorted_highdim_edge_grid(g_h, g_l, mg, side)
            mortar_grid, mortar_cells = _sorted_side_grid(mg, g_l, side)
            low_grid, low_cells = _sorted_low_grid(g_l)

            # Merge the three grids into one
            merged_grid = _merge_grids(low_grid, mortar_grid, high_grid)

            # Note that the following mappings are local for each merged grid.
            # For example, to retrieve the global fracture faces indices, we should
            # write frac_faces[merged_high_ele], and to retrieve the global mortar
            # cells, we should write mortar_cells[merged_mortar_ele]
            # Retrieve element mapping from sorted grids to merged grid
            merged_high_ele = _get_grid_uniongrid_elements(merged_grid, high_grid)
            merged_mortar_ele = _get_grid_uniongrid_elements(merged_grid, mortar_grid)
            merged_low_ele = _get_grid_uniongrid_elements(merged_grid, low_grid)

            # Get projected pressure jump, normal permeabilities, and normal velocities
            pressure_jump, k_perp, _ = _project_poly_to_merged_grid(
                self.estimates,
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
                p_jump = utils.eval_P1(pressure_jump, -coors)  # -coors because is rotated
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

    # Exact velocity error
    def velocity_error(self) -> float:

        true_error = (self.velocity_error_squared_2d().sum()
                      + self.velocity_error_squared_1d().sum()
                      + self.velocity_error_squared_mortar().sum()
                      ) ** 0.5

        return true_error

    def velocity_error_squared_2d(self) -> np.ndarray:

        # Get hold of numerical velocities and create list of coefficients
        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g2d, self.g2d)

        # Compute the true error for each subregion
        integral = np.zeros(self.g2d.num_cells)

        for (vel, idx) in zip(self.u2d("fun"), self.cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x):
                vel_exact_x = vel[0](x[0], x[1])
                vel_exact_y = vel[1](x[0], x[1])

                vel_recon_x = u[0] * x[0] + u[1]
                vel_recon_y = u[0] * x[1] + u[2]

                int_x = (vel_exact_x - vel_recon_x) ** 2
                int_y = (vel_exact_y - vel_recon_y) ** 2

                return int_x + int_y

            integral += method.integrate(integrand, elements) * idx

        return integral

    def velocity_error_squared_1d(self) -> np.ndarray:

        # Get hold of approximated velocities and create list of coefficients
        recon_u = self.d1d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.c1.newton_cotes_closed(10)
        g_rot = utils.rotate_embedded_grid(self.g1d)
        elements = utils.get_quadpy_elements(self.g1d, g_rot)

        # Compute the true error
        def integrand(x):
            # -x to use physical coordinates, and -self. to account for the rotation
            vel_exact_x = -self.u1d("fun")(-x)
            vel_recon_x = u[0] * x + u[1]
            int_x = (vel_exact_x - vel_recon_x) ** 2
            return int_x

        integral = method.integrate(integrand, elements)

        return integral

    def velocity_error_squared_mortar(self) -> np.ndarray:

        # Import functions
        from mdestimates._error_evaluation import _sorted_side_grid, _get_normal_velocity

        # Get hold of grids and dictionaries
        g_l, g_h = self.gb.nodes_of_edge(self.e)
        mg = self.mg

        # Get hold normal velocities
        normal_vel = _get_normal_velocity(self.estimates, self.de)

        # Loop over the sides of the mortar grid
        true_error = np.zeros(mg.num_cells)

        # Obtain the number of sides of the mortar grid
        num_sides = mg.num_sides()
        if num_sides == 2:
            sides = [-1, 1]
        else:
            sides = [1]

        for side in sides:

            # Get rotated grids and sorted elements
            mortar_grid, mortar_cells = _sorted_side_grid(mg, g_l, side)

            # Retrieve normal velocities from the side grid
            normal_vel_side = normal_vel[mortar_cells]

            # Define integration method and obtain quadpy elements
            method = qp.c1.newton_cotes_closed(10)
            qp_ele = utils.get_qp_elements_from_union_grid_1d(mortar_grid)
            qp_ele *= -1  # To use real coordinates

            # Define integrand
            def integrand(y):
                lmbda_ex = self.lmbda("fun")(y)
                lmbda_num = normal_vel_side
                return (lmbda_ex - lmbda_num) ** 2

            # Evaluate integral
            side_error = method.integrate(integrand, qp_ele)

            # Append into the list
            true_error[mortar_cells] = side_error

        return true_error

