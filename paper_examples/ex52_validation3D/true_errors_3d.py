import mdestimates as mde
import porepy as pp
import numpy as np
import quadpy as qp
import scipy.sparse as sps

import mdestimates.estimates_utils as utils
from mdestimates._velocity_reconstruction import (
    _internal_source_term_contribution as mortar_jump,
)
from analytical_3d import ExactSolution3D


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
        g_rot = utils.rotate_embedded_grid(self.g2d)
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
        g_rot = utils.rotate_embedded_grid(self.g2d)
        cc = g_rot.cell_centers

        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_1d_x = u[0].flatten() * cc[0] + u[1].flatten()
        ru_cc_1d_y = u[0].flatten() * cc[1] + u[2].flatten()

        ru_cc_1d = np.array([ru_cc_1d_x, ru_cc_1d_y])

        return ru_cc_1d

    # Residual errors
    def residual_error_local_poincare(self) -> float:
        """Global matrix residual error using local Poincare constants"""

        residual_error_3d = np.sum(self.residual_error_3d_local_poincare()) ** 0.5

        residual_error_2d = np.sum(self.residual_error_2d_local_poincare()) ** 0.5

        residual_error = residual_error_3d + residual_error_2d

        return residual_error

    def residual_error_global_poincare(self) -> float:
        """Global matrix residual error using global Poincare constants"""

        residual_error_3d = np.sum(self.residual_error_3d_global_poincare()) ** 0.5

        residual_error_2d = np.sum(self.residual_error_2d_global_poincare()) ** 0.5

        residual_error = residual_error_2d + residual_error_3d

        return residual_error

    def residual_error_3d_local_poincare(self) -> np.ndarray:
        """Local matrix residual error squared for local Poincare constants"""

        weight_3d = (self.g3d.cell_diameters() / np.pi) ** 2
        residual_norm = self.residual_norm_squared_3d()

        residual_error_3d = weight_3d * residual_norm

        return residual_error_3d

    def residual_error_3d_global_poincare(self) -> np.ndarray:
        """Local matrix residual error squared for global Poincare constants"""

        weight = (1 / (np.sqrt(3) * np.pi)) ** 2
        residual_norm = self.residual_norm_squared_3d()

        residual_error_3d = weight * residual_norm

        return residual_error_3d

    def residual_error_2d_local_poincare(self) -> np.ndarray:
        """Local fracture residual error squared for local Poincare constant"""

        weight_2d = (self.g2d.cell_diameters() / np.pi) ** 2
        residual_norm = self.residual_norm_squared_2d()

        residual_error_1d = weight_2d * residual_norm

        return residual_error_1d

    def residual_error_2d_global_poincare(self) -> np.ndarray:
        """Local fracture residual error squared for global Poincare const"""

        weight = (1 / (np.sqrt(3) * np.pi)) ** 2
        residual_norm = self.residual_norm_squared_2d()

        residual_error_2d = weight * residual_norm

        return residual_error_2d

    def residual_norm_squared_3d(self) -> np.ndarray:
        """Compute square of residual errors for 2D (onnly the norm)"""

        # Retrieve reconstructed velocity
        recon_u = self.d3d[self.estimates.estimates_kw]["recon_u"].copy()

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

    def residual_norm_squared_2d(self) -> np.ndarray:
        """Compute square of residual errors for 1D (onnly the norm)"""

        # Retrieve reconstructed velocity
        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Divergence of the reconstructed flux
        div_u = 2 * u[0]

        # Jump in mortar fluxes
        jump_in_mortars = (
            mortar_jump(self.estimates, self.g2d) / self.g2d.cell_volumes
        ).reshape(self.g2d.num_cells, 1)

        # Integration method and retrieving elements
        method = qp.t2.get_good_scheme(10)
        g_rot = utils.rotate_embedded_grid(self.g2d)
        elements = utils.get_quadpy_elements(self.g2d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        def integrand(x):
            return (self.f2d("fun")(x[0], x[1]) - div_u + jump_in_mortars) ** 2

        integral = method.integrate(integrand, elements)

        return integral

    # Exact pressure error
    def pressure_error(self) -> float:

        true_error = (
            self.pressure_error_squared_3d().sum()
            + self.pressure_error_squared_2d().sum()
            + self.pressure_error_squared_mortar().sum()
        ) ** 0.5

        return true_error

    def pressure_error_squared_3d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coefficients
        recon_p = self.d3d[self.estimates.estimates_kw]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(5)
        elements = utils.get_quadpy_elements(self.g3d, self.g3d)

        # Compute the true error for each subregion
        integral = np.zeros(self.g3d.num_cells)

        # Compute integrals
        for (gradp, idx) in zip(self.gradp3d("fun"), self.cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x):
                # gradp exact in x and y
                gradp_exact_x = gradp[0](x[0], x[1], x[2])
                gradp_exact_y = gradp[1](x[0], x[1], x[2])
                gradp_exact_z = gradp[2](x[0], x[1], x[2])
                # gradp reconstructed in x, y, and z
                gradp_recon_x = pr[0] * np.ones_like(x[0])
                gradp_recon_y = pr[1] * np.ones_like(x[1])
                gradp_recon_z = pr[2] * np.ones_like(x[2])
                # integral in x and y
                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2
                int_z = (gradp_exact_z - gradp_recon_z) ** 2
                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def pressure_error_squared_2d(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = self.d2d[self.estimates.estimates_kw]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(10)
        g_rot = utils.rotate_embedded_grid(self.g2d)
        elements = utils.get_quadpy_elements(self.g2d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        # Compute the true error
        def integrand(x):
            # Exact pressure gradient
            gradp_exact_rot_y = -self.gradp2d("fun")[0](
                x[0], x[1]
            )  # due to rotation (hardcoded)
            gradp_exact_rot_z = -self.gradp2d("fun")[1](
                x[0], x[1]
            )  # due to rotation (hardcoded)
            # Reconstructed pressure gradient
            gradp_recon_y = pr[0] * np.ones_like(x[0])
            gradp_recon_z = pr[1] * np.ones_like(x[1])
            # integral
            int_y = (gradp_exact_rot_y - gradp_recon_y) ** 2
            int_z = (gradp_exact_rot_z - gradp_recon_z) ** 2
            return int_y + int_z

        integral = method.integrate(integrand, elements)

        return integral

    def pressure_error_squared_mortar(self) -> np.ndarray:

        # Import functions
        from mdestimates._error_evaluation import (
            _get_high_pressure_trace,
            _get_low_pressure,
        )

        def compute_sidegrid_error(side_tuple):
            """
            This functions projects a mortar quantity to the side grids, and
            then performs the integration on the given side grid.

            Parameters
            ----------
            side_tuple : Tuple
                Containing the sidegrids

            Returns
            -------
            true_error_side : NumPy nd-Array of size (sidegrid.num_cells, 1)
                True error (squared) for each element of the side grid.

            """

            # Get projector and sidegrid object
            projector = side_tuple[0]
            sidegrid = side_tuple[1]

            # Rotate side-grid
            sidegrid_rot = utils.rotate_embedded_grid(sidegrid)

            # Obtain QuadPy elements
            elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)
            elements *= -1  # to use real coordinates

            # Project relevant quantities to the side grids
            deltap_side = projector * deltap

            # Declare integrand
            def integrand(x):
                true_jump = self.p2d("fun")
                p_jump = utils.eval_P1(deltap_side, -x)  # -x due to rotation
                diff = true_jump(x[0], x[1]) - p_jump
                return diff ** 2

            # Compute integral
            true_error_side = method.integrate(integrand, elements)

            return true_error_side

        # Get hold of mortar grid and check the dimensionality
        mg = self.mg

        # Obtain higher- and lower-dimensional grids and dictionaries
        g_l, g_h = self.gb.nodes_of_edge(self.e)
        d_h = self.gb.node_props(g_h)
        d_l = self.gb.node_props(g_l)

        # Face-cell map between higher- and lower-dimensional subdomains
        frac_faces = sps.find(mg.primary_to_mortar_avg().T)[0]
        frac_cells = sps.find(mg.secondary_to_mortar_avg().T)[0]

        # Obtain the trace of the higher-dimensional pressure
        tracep_high = _get_high_pressure_trace(
            self.estimates, g_l, g_h, d_h, frac_faces
        )

        # Obtain the lower-dimensional pressure
        p_low = _get_low_pressure(self.estimates, g_l, d_l, frac_cells)

        # Now, we can work with the pressure difference
        deltap = p_low - tracep_high

        # Declare integration method
        method = qp.t2.get_good_scheme(10)

        # Retrieve side-grids tuples
        sides = mg.project_to_side_grids()

        # Compute the errors for each sidegrid
        mortar_error = []
        for side in sides:
            mortar_error.append(compute_sidegrid_error(side))

        # Concatenate into one numpy array
        true_error_mortar = np.concatenate(mortar_error)

        return true_error_mortar

    # Exact velocity error
    def velocity_error(self) -> float:

        true_error = (
            self.velocity_error_squared_3d().sum()
            + self.velocity_error_squared_2d().sum()
            + self.velocity_error_squared_mortar().sum()
        ) ** 0.5

        return true_error

    def velocity_error_squared_3d(self) -> np.ndarray:

        # Get hold of numerical velocities and create list of coefficients
        recon_u = self.d3d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(5)
        elements = utils.get_quadpy_elements(self.g3d, self.g3d)

        # Compute the true error for each subregion
        integral = np.zeros(self.g3d.num_cells)

        for (vel, idx) in zip(self.u3d("fun"), self.cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x):
                vel_exact_x = vel[0](x[0], x[1], x[2])
                vel_exact_y = vel[1](x[0], x[1], x[2])
                vel_exact_z = vel[2](x[0], x[1], x[2])

                vel_recon_x = u[0] * x[0] + u[1]
                vel_recon_y = u[0] * x[1] + u[2]
                vel_recon_z = u[0] * x[2] + u[3]

                int_x = (vel_exact_x - vel_recon_x) ** 2
                int_y = (vel_exact_y - vel_recon_y) ** 2
                int_z = (vel_exact_z - vel_recon_z) ** 2

                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def velocity_error_squared_2d(self) -> np.ndarray:

        # Get hold of approximated velocities and create list of coefficients
        recon_u = self.d2d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(10)
        g_rot = utils.rotate_embedded_grid(self.g2d)
        elements = utils.get_quadpy_elements(self.g2d, g_rot)

        # Compute the true error
        def integrand(x):
            # -x to use physical coordinates, and -self. to account for
            # the rotation
            vel_exact_y = -self.u2d("fun")[0](-x[0], -x[1])
            vel_exact_z = -self.u2d("fun")[1](-x[0], -x[1])
            vel_recon_y = u[0] * x[0] + u[1]
            vel_recon_z = u[0] * x[1] + u[2]
            int_y = (vel_exact_y - vel_recon_y) ** 2
            int_z = (vel_exact_z - vel_recon_z) ** 2
            return int_y + int_z

        integral = method.integrate(integrand, elements)

        return integral

    def velocity_error_squared_mortar(self) -> np.ndarray:

        # Import functions
        from mdestimates._error_evaluation import _get_normal_velocity

        def compute_sidegrid_error(side_tuple):

            # Get projector and sidegrid object
            projector = side_tuple[0]
            sidegrid = side_tuple[1]

            # Rotate side-grid
            sidegrid_rot = utils.rotate_embedded_grid(sidegrid)

            # Obtain QuadPy elements
            elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)

            # Project relevant quanitites to the side grid
            normalvel_side = projector * normal_vel

            # Declare integrand
            def integrand(x):
                lmbda_ex = self.lmbda("fun")(x[0], x[1])
                lmbda_num = normalvel_side
                return (lmbda_ex - lmbda_num) ** 2

            # Compute integral
            diffusive_error_side = method.integrate(integrand, elements)

            return diffusive_error_side

        # Get mortar grid and check dimensionality
        mg = self.mg

        # Obtain normal velocities
        normal_vel = _get_normal_velocity(self.estimates, self.de)

        # Declare integration method
        method = qp.t2.get_good_scheme(5)

        # Retrieve side-grids tuples
        sides = mg.project_to_side_grids()

        # Compute the errors for each sidegrid
        true_error_side = []
        for side in sides:
            true_error_side.append(compute_sidegrid_error(side))

        # Concatenate into one numpy array
        true_error = np.concatenate(true_error_side)

        return true_error

    # Exact combined error
    def combined_error_local_poincare(self) -> float:
        """Computes combined error using local Poincare constants"""

        combined_error = (
            self.pressure_error()
            + self.velocity_error()
            + self.residual_error_local_poincare()
        )

        return combined_error

    def combined_error_global_poincare(self) -> float:
        """Computes combined error using the global Poincare constant"""

        combined_error = (
            self.pressure_error()
            + self.velocity_error()
            + self.residual_error_global_poincare()
        )

        return combined_error
