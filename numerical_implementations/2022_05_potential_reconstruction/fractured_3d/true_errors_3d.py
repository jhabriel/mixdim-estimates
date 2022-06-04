import mdestimates as mde
import porepy as pp
import numpy as np
import quadpy as qp
import scipy.sparse as sps

import mdestimates.estimates_utils as utils
from analytical_3d import ExactSolution3D

from typing import Tuple


class TrueErrors3D(ExactSolution3D):
    def __init__(self, gb: pp.GridBucket, estimates: mde.ErrorEstimate):
        super().__init__(gb)
        self.estimates = estimates

    def __repr__(self) -> str:
        return "True error object for 3D validation"

    # Reconstructed pressures
    def reconstructed_p3d(self) -> np.ndarray:
        """Compute reconstructed pressure for the matrix"""

        cc = self.g3d.cell_centers

        # Get reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        rp_cc_3d = np.zeros([self.g3d.num_cells, 1])
        for idx in self.cell_idx:
            rp_cc_3d += (
                p[0] * cc[0].reshape(self.g3d.num_cells, 1)
                + p[1] * cc[1].reshape(self.g3d.num_cells, 1)
                + p[2] * cc[2].reshape(self.g3d.num_cells, 1)
                + p[3]
            ) * idx.reshape(self.g3d.num_cells, 1)

        return rp_cc_3d.flatten()

    def reconstructed_p2d(self) -> np.ndarray:
        """Compute reconstructed pressure for the fracture"""

        # Get hold of reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Rotate grid and get cell centers
        g_rot = mde.RotatedGrid(self.g2d)
        cc = g_rot.cell_centers

        # Project reconstructed pressure onto the cell centers
        rp_cc_2d = (
            p[0] * cc[0].reshape(self.g2d.num_cells, 1)
            + p[1] * cc[1].reshape(self.g2d.num_cells, 1)
            + p[2]
        )

        return rp_cc_2d.flatten()

    # Reconstructed pressure gradients
    def reconstructed_gradp3(self) -> np.ndarray:
        """Compute reconstructed pressure gradient for the matrix"""

        # Get reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpx_cc_2d = p[0]
        gradpy_cc_2d = p[1]
        gradpz_cc_2d = p[2]

        gradp_cc = np.array(
            [gradpx_cc_2d.flatten(), gradpy_cc_2d.flatten(), gradpz_cc_2d.flatten(),]
        )

        return gradp_cc

    def reconstructed_gradp2(self) -> np.ndarray:
        """Compute reconstructed pressure gradient for the fracture"""

        # Get reconstructed pressure
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpy_cc_2d = p[0]
        gradpz_cc_2d = p[1]

        gradp_cc = np.array([gradpy_cc_2d.flatten(), gradpz_cc_2d.flatten(),])

        return gradp_cc

    # Reconstructed velocities
    def reconstructed_u3d(self) -> np.ndarray:
        """Compute reconstructed velocity for the matrix"""

        cc = self.g3d.cell_centers

        recon_u = self.d3d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_3d_x = np.zeros([self.g3d.num_cells, 1])
        ru_cc_3d_y = np.zeros([self.g3d.num_cells, 1])
        ru_cc_3d_z = np.zeros([self.g3d.num_cells, 1])

        for idx in self.cell_idx:
            ru_cc_3d_x += (
                u[0] * cc[0].reshape(self.g3d.num_cells, 1) + u[1]
            ) * idx.reshape(self.g3d.num_cells, 1)
            ru_cc_3d_y += (
                u[0] * cc[1].reshape(self.g3d.num_cells, 1) + u[2]
            ) * idx.reshape(self.g3d.num_cells, 1)
            ru_cc_3d_z += (
                u[0] * cc[2].reshape(self.g3d.num_cells, 1) + u[3]
            ) * idx.reshape(self.g3d.num_cells, 1)

        ru_cc_2d = np.array(
            [ru_cc_3d_x.flatten(), ru_cc_3d_y.flatten(), ru_cc_3d_z.flatten(),]
        )

        return ru_cc_2d

    def reconstructed_u2d(self) -> np.ndarray:
        """Compute reconstructed velocity for the fracture"""

        # Rotate embedded grid and
        g_rot = mde.RotatedGrid(self.g2d)
        cc = g_rot.cell_centers

        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_1d_x = u[0].flatten() * cc[0] + u[1].flatten()
        ru_cc_1d_y = u[0].flatten() * cc[1] + u[2].flatten()

        ru_cc_1d = np.array([ru_cc_1d_x, ru_cc_1d_y])

        return ru_cc_1d

    # Residual errors
    def residual_error(self) -> float:
        """Global matrix residual error using local Poincare constants"""

        residual_error_3d = np.sum(self.residual_error_3d()) ** 0.5

        residual_error_2d = np.sum(self.residual_error_2d()) ** 0.5

        residual_error = residual_error_3d + residual_error_2d

        return residual_error

    def residual_error_3d(self) -> np.ndarray:
        """Local matrix residual error squared for local Poincare constants"""

        weight_3d = (self.g3d.cell_diameters() / np.pi) ** 2
        residual_norm = self.residual_norm_sq_3d()

        residual_error_3d = weight_3d * residual_norm

        return residual_error_3d

    def residual_error_2d(self) -> np.ndarray:
        """Local fracture residual error squared for local Poincare constant"""

        weight_2d = (self.g2d.cell_diameters() / np.pi) ** 2
        residual_norm = self.residual_norm_sq_2d()

        residual_error_1d = weight_2d * residual_norm

        return residual_error_1d

    def residual_norm_sq_3d(self) -> np.ndarray:
        """Compute square of residual errors for 2D (onnly the norm)"""

        # Retrieve reconstructed velocity
        recon_u = self.d3d[self.estimates.estimates_kw]["recon_u"]

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Divergence of the reconstructed flux
        div_u = 3 * u[0]

        # Integration method and retrieving elements
        int_method = qp.t3.get_good_scheme(5)
        elements = utils.get_quadpy_elements(self.g3d, self.g3d)

        integral = np.zeros(self.g3d.num_cells)
        for (f, idx) in zip(self.f3d("fun"), self.cell_idx):
            # Declare integrand
            def integrand(x):
                return (f(x[0], x[1], x[2]) * np.ones_like(x[0]) - div_u) ** 2

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    def residual_norm_sq_2d(self) -> np.ndarray:
        """Compute square of residual errors for 1D (onnly the norm)"""

        # Retrieve reconstructed velocity
        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"]

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Divergence of the reconstructed flux
        div_u = 2 * u[0]

        # Jump in mortar fluxes
        jump_in_mortars = (
            self.d2d[self.estimates.estimates_kw]["mortar_jump"].copy() / self.g2d.cell_volumes
        ).reshape(self.g2d.num_cells, 1)

        # Integration method and retrieving elements
        method = qp.t2.get_good_scheme(10)
        g_rot = mde.RotatedGrid(self.g2d)
        elements = utils.get_quadpy_elements(self.g2d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        def integrand(x):
            return (self.f2d("fun")(x[0], x[1]) - div_u + jump_in_mortars) ** 2

        integral = method.integrate(integrand, elements)

        return integral

    # Exact pressure error
    def pressure_error(self) -> float:

        true_error = np.sqrt(
            self.pressure_error_sq_3d().sum()
            + self.pressure_error_sq_2d().sum()
            + self.pressure_error_sq_mortar().sum()
        )

        return true_error

    def pressure_error_sq_3d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coefficients
        recon_p = self.d3d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g3d, self.g3d)

        # Compute the true error for each subregion
        integral = np.zeros(self.g3d.num_cells)

        # Compute integrals
        for (gradp, idx) in zip(self.gradp3d("fun"), self.cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x):
                # gradp exact in x, y and z
                gradp_exact_x = gradp[0](x[0], x[1], x[2])
                gradp_exact_y = gradp[1](x[0], x[1], x[2])
                gradp_exact_z = gradp[2](x[0], x[1], x[2])
                if self.estimates.p_degree == 1:  # P1 elements
                    # gradp reconstructed in x, y, and z
                    gradp_recon_x = p[0] * np.ones_like(x[0])
                    gradp_recon_y = p[1] * np.ones_like(x[1])
                    gradp_recon_z = p[2] * np.ones_like(x[2])
                else:  # P2 elements
                    # gradp reconstructed in x, y, and z
                    gradp_recon_x = (
                            2 * p[0] * x[0]
                            + p[1] * x[1]
                            + p[2] * x[2]
                            + p[3] * np.ones_like(x[0])
                    )
                    gradp_recon_y = (
                            p[1] * x[0]
                            + 2 * p[4] * x[1]
                            + p[5] * x[2]
                            + p[6] * np.ones_like(x[1])
                    )
                    gradp_recon_z = (
                            p[2] * x[0]
                            + p[5] * x[1]
                            + 2 * p[7] * x[2]
                            + p[8] * np.ones_like(x[2])
                    )
                # integral in x, y, and z
                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2
                int_z = (gradp_exact_z - gradp_recon_z) ** 2
                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def pressure_error_sq_2d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(10)
        g_rot = mde.RotatedGrid(self.g2d)
        elements = utils.get_quadpy_elements(self.g2d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        # Compute the true error
        def integrand(x):
            # Exact pressure gradient
            gradp_exact_rot_y = -self.gradp2d("fun")[0](x[0], x[1])  # due to rot (hardcoded)
            gradp_exact_rot_z = -self.gradp2d("fun")[1](x[0], x[1])  # due to rot (hardcoded)
            if self.estimates.p_degree == 1:  # P1 elements
                gradp_recon_y = p[0] * np.ones_like(x[0])
                gradp_recon_z = p[1] * np.ones_like(x[1])
            else:  # P2 elements
                gradp_recon_y = 2 * p[0] * -x[0] + p[1] * -x[1] + p[2] * np.ones_like(-x[0])
                gradp_recon_z = p[1] * -x[0] + 2 * p[3] * -x[1] + p[4] * np.ones_like(-x[1])
            # Compute integrals
            int_y = (gradp_exact_rot_y - gradp_recon_y) ** 2
            int_z = (gradp_exact_rot_z - gradp_recon_z) ** 2
            return int_y + int_z

        integral = method.integrate(integrand, elements)

        return integral

    def pressure_error_sq_mortar(self) -> np.ndarray:

        dfe = mde.DiffusiveError(self.estimates)

        def compute_sidegrid_error(
                side_tuple: Tuple,
                delta_pressure,
                exact_pressure,
                pressure_degree,
                integration_method,
                normal_perm
        ) -> np.ndarray:
            """
            Projects a mortar quantity to a side grid and perform numerical integration.

            Parameters
            ----------
                side_tuple (Tuple): Containing the sidegrids

            Returns
            -------
                diffusive_error_side (np.ndarray): Diffusive error (squared) for each element
                    of the side grid.
            """

            # Get projector and sidegrid object
            projector = side_tuple[0]
            sidegrid = side_tuple[1]

            # Rotate side-grid
            sidegrid_rot = mde.RotatedGrid(sidegrid)

            # Obtain QuadPy elements
            elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)
            elements *= -1  # to use real coordinates

            # Project relevant quantities to the side grids
            deltap_side = projector * delta_pressure

            # Project normal permeability
            k_side = projector * normal_perm

            # Declare integrand
            def integrand(x):
                if pressure_degree == 1:
                    p_jump = utils.eval_p1(deltap_side, -x)  # -x due to rotation
                else:
                    p_jump = utils.eval_p2(deltap_side, -x)  # -x due to rotation
                return (k_side ** 0.5 * (exact_pressure(x[0], x[1]) - p_jump)) ** 2

            # Compute integral
            true_error_side = integration_method.integrate(integrand, elements)

            return true_error_side

        # Get mortar grid and check dimensionality
        mg = self.mg
        if mg.dim != 2:
            raise ValueError("Expected two-dimensional mortar grid")

        # Get hold of higher- and lower-dimensional neighbors and their dictionaries
        g_l, g_h = self.gb.nodes_of_edge(self.e)
        d_h = self.gb.node_props(g_h)
        d_l = self.gb.node_props(g_l)

        # Retrieve normal diffusivity
        normal_diff = self.de[pp.PARAMETERS][self.estimates.kw]["normal_diffusivity"]
        if isinstance(normal_diff, int) or isinstance(normal_diff, float):
            k = normal_diff * np.ones([mg.num_cells, 1])
        else:
            k = normal_diff.reshape(mg.num_cells, 1)

        # Face-cell map between higher- and lower-dimensional subdomains
        frac_faces = sps.find(mg.primary_to_mortar_avg().T)[0]
        frac_cells = sps.find(mg.secondary_to_mortar_avg().T)[0]

        # Obtain the trace of the higher-dimensional pressure
        tracep_high = dfe._get_high_pressure_trace(g_l, g_h, d_h, frac_faces)

        # Obtain the lower-dimensional pressure
        p_low = dfe._get_low_pressure(d_l, frac_cells)

        # Now, we can work with the pressure difference
        deltap = p_low - tracep_high

        # Declare integration method
        method = qp.t2.get_good_scheme(10)

        # Retrieve side-grids tuples
        sides = mg.project_to_side_grids()

        # Compute the errors for each sidegrid
        diffusive = []
        for side in sides:
            diffusive.append(
                compute_sidegrid_error(
                    side,
                    deltap,
                    self.p2d("fun"),
                    self.estimates.p_degree,
                    method,
                    k
                )
            )

        # Concatenate into one numpy array
        diffusive_error = np.concatenate(diffusive)

        return diffusive_error

