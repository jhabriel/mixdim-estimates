import mdestimates as mde
import porepy as pp
import numpy as np
import quadpy as qp

import mdestimates.estimates_utils as utils

from analytical import ExactSolution

from typing import List


class TrueError(ExactSolution):
    def __init__(self, estimates: mde.ErrorEstimate):
        super().__init__(gb=estimates.gb)
        self.estimates = estimates

    def __repr__(self) -> str:
        return "True error object for 3D validation"

    # Reconstructed pressures
    def reconstructed_p_p1(self) -> np.ndarray:
        """Compute reconstructed pressure for P1 elements"""

        cc = self.g.cell_centers

        # Get reconstructed pressure
        recon_p = self.d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        rp_cc = (p[0] * cc[0].reshape(self.g.num_cells, 1)
                 + p[1] * cc[1].reshape(self.g.num_cells, 1)
                 + p[2] * cc[2].reshape(self.g.num_cells, 1)
                 + p[3]
                 )

        return rp_cc.flatten()

    def reconstructed_p_p2(self) -> np.ndarray:
        """Compute reconstructed pressure for P2 elements"""

        cc = self.g.cell_centers
        nc = self.g.num_cells

        # Get reconstructed pressure
        recon_p = self.d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)
        x = cc[0].reshape(nc, 1)
        y = cc[1].reshape(nc, 1)
        z = cc[2].reshape(nc, 1)

        # Project reconstructed pressure onto the cell centers
        rp_cc = (p[0] * x ** 2
                 + p[1] * x * y
                 + p[2] * x * z
                 + p[3] * x
                 + p[4] * y ** 2
                 + p[5] * y * z
                 + p[6] * y
                 + p[7] * z ** 2
                 + p[8] * z
                 + p[9]
                 )

        return rp_cc.flatten()

    def postprocessed_p_p2(self) -> np.ndarray:
        """Compute postprocessed pressure for P2 elements"""

        cc = self.g.cell_centers
        nc = self.g.num_cells

        # Get reconstructed pressure
        recon_p = self.d[self.estimates.estimates_kw]["postprocessed_p"]
        p = utils.poly2col(recon_p)
        x = cc[0].reshape(nc, 1)
        y = cc[1].reshape(nc, 1)
        z = cc[2].reshape(nc, 1)

        # Project reconstructed pressure onto the cell centers
        rp_cc = (p[0] * x ** 2
                 + p[1] * x * y
                 + p[2] * x * z
                 + p[3] * x
                 + p[4] * y ** 2
                 + p[5] * y * z
                 + p[6] * y
                 + p[7] * z ** 2
                 + p[8] * z
                 + p[9]
                 )

        return rp_cc.flatten()


    # Reconstructed pressure gradients
    def reconstructed_gradp_p1(self) -> List[np.ndarray]:
        """Compute reconstructed pressure gradient for P1 elements"""

        # Get reconstructed pressure
        recon_p = self.d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpx_cc = p[0]
        gradpy_cc = p[1]
        gradpz_cc = p[2]

        gradp_cc = [
            gradpx_cc.flatten(),
            gradpy_cc.flatten(),
            gradpz_cc.flatten(),
        ]

        return gradp_cc

    # Reconstructed velocities
    def reconstructed_u(self) -> List[np.ndarray]:
        """Compute reconstructed velocity for the matrix"""

        cc = self.g.cell_centers

        recon_u = self.d[self.estimates.estimates_kw]["recon_u"]
        u = utils.poly2col(recon_u)

        ru_cc_x = u[0] * cc[0].reshape(self.g.num_cells, 1) + u[1]
        ru_cc_y = u[0] * cc[1].reshape(self.g.num_cells, 1) + u[2]
        ru_cc_z = u[0] * cc[2].reshape(self.g.num_cells, 1) + u[3]

        ru_cc = [
            ru_cc_x.flatten(),
            ru_cc_y.flatten(),
            ru_cc_z.flatten(),
        ]

        return ru_cc

    # Residual errors
    def residual_error(self) -> float:
        """Residual error using local Poincare constants"""

        # Retrieve reconstructed velocity
        recon_u = self.d[self.estimates.estimates_kw]["recon_u"]

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Divergence of the reconstructed flux
        div_u = 3 * u[0]

        # Integration method and retrieving elements
        int_method = qp.t3.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g, self.g)

        # Declare integrand
        def integrand(x):
            source = self.f("fun")(x[0], x[1], x[2]) * np.ones_like(x[0])
            out = (source - div_u) ** 2
            return out

        # Compute numerical integration
        residual_norm_squared = int_method.integrate(integrand, elements)

        # Scale with weights
        weights = (self.g.cell_diameters() / np.pi) ** 2
        residual_error_squared = weights * residual_norm_squared

        # Sum and take the square root
        residual_error = residual_error_squared.sum() ** 0.5

        return residual_error

    # Exact pressure error
    def true_error(self) -> float:

        # Get hold of reconstructed pressure and create list of coefficients
        if self.estimates.p_recon_method in ["keilegavlen", "cochez"]:
            p = self.d[self.estimates.estimates_kw]["recon_p"]
        else:
            p = self.d[self.estimates.estimates_kw]["postprocessed_p"]

        p = utils.poly2col(p)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g, self.g)

        # Declare integrand
        def integrand(x):
            # gradp exact in x and y
            gradp_exact_x = self.gradp("fun")[0](x[0], x[1], x[2])
            gradp_exact_y = self.gradp("fun")[1](x[0], x[1], x[2])
            gradp_exact_z = self.gradp("fun")[2](x[0], x[1], x[2])

            # gradp reconstructed in x, y, and z
            # Recall that:
            # p(x,y,z)|K = c0x^2 + c1xy + c2xz + c3x + c4y^2 + c5yz + c6y + c7z^2 + c8z + c9
            #
            #                  [ 2c0x + c1y + c2z + c3 ]
            # gradp(x,y,z)|K = [ c1x + 2c4y + c5z + c6 ]
            #                  [ c2x + c5y + 2c7z + c8 ]
            #
            if self.estimates.p_recon_method in ["keilegavlen", "cochez"]:
                gradp_recon_x = p[0] * np.ones_like(x[0])
                gradp_recon_y = p[1] * np.ones_like(x[1])
                gradp_recon_z = p[2] * np.ones_like(x[2])
            else:
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

        # Compute numerical integration
        integral = method.integrate(integrand, elements)

        # Compute true error
        true_error = integral.sum() ** 0.5

        return true_error
