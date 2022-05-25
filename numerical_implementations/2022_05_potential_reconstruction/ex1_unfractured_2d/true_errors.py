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
        return "True error object for 2D validation"

    # Reconstructed pressures
    def reconstructed_p(self) -> np.ndarray:
        """Compute reconstructed pressure"""

        cc = self.g.cell_centers

        # Get reconstructed pressure
        recon_p = self.d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        rp_cc = (p[0] * cc[0].reshape(self.g.num_cells, 1)
                 + p[1] * cc[1].reshape(self.g.num_cells, 1)
                 + p[2]
                 )

        return rp_cc.flatten()

    # Reconstructed pressure gradients
    def reconstructed_gradp(self) -> List[np.ndarray]:
        """Compute reconstructed pressure gradient"""

        # Get reconstructed pressure
        recon_p = self.d[self.estimates.estimates_kw]["recon_p"]
        p = utils.poly2col(recon_p)

        # Project reconstructed pressure onto the cell centers
        gradpx_cc = p[0]
        gradpy_cc = p[1]
        gradpz_cc = np.zeros([self.g.num_cells, 1])

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

        recon_u = self.d[self.estimates.estimates_kw]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        ru_cc_x = u[0] * cc[0].reshape(self.g.num_cells, 1) + u[1]
        ru_cc_y = u[0] * cc[1].reshape(self.g.num_cells, 1) + u[2]
        ru_cc_z = np.zeros([self.g.num_cells, 1])

        ru_cc_2d = [
            ru_cc_x.flatten(),
            ru_cc_y.flatten(),
            ru_cc_z.flatten(),
        ]

        return ru_cc_2d

    # Residual errors
    def residual_error(self) -> float:
        """Residual error using local Poincare constants"""

        # Retrieve reconstructed velocity
        recon_u = self.d[self.estimates.estimates_kw]["recon_u"]

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)

        # Divergence of the reconstructed flux
        div_u = 2 * u[0]

        # Integration method and retrieving elements
        int_method = qp.t2.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g, self.g)

        # Declare integrand
        def integrand(x):
            source = self.f("fun")(x[0], x[1]) * np.ones_like(x[0])
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

        if self.estimates.p_recon_method == "keilegavlen" or "cochez-dhondt":
            true_error = self.pressure_error_squared_p1().sum() ** 0.5
        elif self.estimates.p_recon_method == "vohralik":
            true_error = self.pressure_error_squared_p2().sum() ** 0.5
        else:
            raise NotImplementedError("Pressure reconstruction technique not "
                                      "implemented")

        return true_error

    def pressure_error_squared_p1(self) -> np.ndarray:

        # Get hold of reconstructed pressure and create list of coefficients
        recon_p = self.d[self.estimates.estimates_kw]["recon_p"]
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g, self.g)

        # Declare integrand
        def integrand(x):
            # gradp exact in x and y
            gradp_exact_x = self.gradp("fun")[0](x[0], x[1])
            gradp_exact_y = self.gradp("fun")[1](x[0], x[1])
            # gradp reconstructed in x and y
            gradp_recon_x = pr[0] * np.ones_like(x[0])
            gradp_recon_y = pr[1] * np.ones_like(x[1])
            # integral in x and y
            int_x = (gradp_exact_x - gradp_recon_x) ** 2
            int_y = (gradp_exact_y - gradp_recon_y) ** 2
            return int_x + int_y

        # Compute numerical integration
        integral = method.integrate(integrand, elements)

        return integral

    def pressure_error_squared_p2(self) -> np.ndarray:

        pass