from __future__ import annotations
import mdestimates as mde
import porepy as pp
import numpy as np
import mdestimates.estimates_utils as utils
import quadpy as qp

from typing import Callable, Union


class ResidualError(mde.ErrorEstimate):
    """ Parent class for the estimation of residual errors. """

    def __init__(self, estimate: mde.ErrorEstimate):
        super().__init__(
            gb=estimate.gb,
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

    def compute_residual_error(self):
        """
        Computes square of the residual error for all nodes of the grid bucket.

        Note:
        -----
            In each data dictionary, the square of the residual error will be stored in
            data[self.estimates_kw]["residual_error"].

        """

        # Loop through all the nodes of the grid bucket
        for node, (g, d) in enumerate(self.gb):

            # Handle the case of zero-dimensional subdomains
            if g.dim == 0:
                d[self.estimates_kw]["residual_error"] = None
                continue

            # Rotate grid. If g == gb.dim_max() this has no effect.
            g_rot: mde.RotatedGrid = mde.RotatedGrid(g)

            # Retrieve source from source list
            source: Union[Callable[..., np.ndarray], int] = self.source_list[node]

            # Obtain the subdomain residual error
            if self.poincare == "global":
                poincare_constant: Union[float, int] = self.poincare_list[node]
                residual_error = self.residual_error_global_poincare(g, g_rot, d, source,
                                                                     poincare_constant)
            else:
                residual_error = self.residual_error_local_poincare(g, g_rot, d, source)

            d[self.estimates_kw]["residual_error"] = residual_error

    def residual_norm_squared(self,
                              g: pp.Grid,
                              g_rot: mde.RotatedGrid,
                              d: dict,
                              source: Union[Callable[..., np.ndarray], int],
                              ) -> np.ndarray:
        """ Computes the square of the residual errors.

        Parameters
        ----------
            g (pp.Grid): PorePy grid.
            g_rot (mde.RotatedGrid): Rotated pseudo-grid.
            d (dict): Data dictionary.
            source (Function or 0): External source term applied to the grid. If 0,
                 then it is assumed that no source term should be imposed.

        Returns
        -------
            residual_norm (np.ndarray): Square of the residual norm for the given grid.

        Note
        ----
            Poincare constants are not included in the calculation. In other words,
            this is only the norm of the residual.

       """

        # Retrieve reconstructed velocity, obtain coefficients, and compute its divergence
        recon_u = d[self.estimates_kw]["recon_u"]
        u = utils.poly2col(recon_u)
        div_u: np.ndarray = g.dim * u[0]

        # Retrieve jump in mortar fluxes
        mortar_jump: np.ndarray = d[self.estimates_kw]["mortar_jump"].reshape(g.num_cells, 1)

        # Integration methods an retrieving elements
        elements = utils.get_quadpy_elements(g, g_rot)
        if g.dim == 1:
            method = qp.c1.newton_cotes_closed(4)
        elif g.dim == 2:
            method = qp.t2.get_good_scheme(4)
        else:
            method = qp.t3.get_good_scheme(4)

        # Perform integration
        def integrand(x: np.ndarray) -> np.ndarray:

            if not isinstance(source, int):
                if g.dim == 3:
                    out = (source(x[0], x[1], x[2]) - div_u + mortar_jump) ** 2
                elif g.dim == 2:
                    out = (source(x[0], x[1]) - div_u + mortar_jump) ** 2
                elif g.dim == 1:
                    out = (source(x[0]) - div_u + mortar_jump) ** 2
            else:
                out = ((-div_u + mortar_jump) * np.ones_like(x[0])) ** 2

            return out

        # Perform numerical integration
        integral = method.integrate(integrand, elements)

        return integral

    def residual_error_global_poincare(self,
                                       g: pp.Grid,
                                       g_rot: mde.RotatedGrid,
                                       d: dict,
                                       source: Union[Callable[..., np.ndarray], int],
                                       poincare_constant: Union[float, int],
                                       ) -> np.ndarray:
        """Residual error squared using the global Poincare constant.

        Parameters
        ----------
            g (pp.Grid): PorePy grid.
            g_rot (mde.RotatedGrid): Rotated pseudo-grid.
            d (dict): Data dictionary.
            source (Function or 0): External source term applied to the grid. If 0,
                 then it is assumed that no source term should be imposed.
            poincare_constant (Integer or Float): Poincare constant for the subdomain.

        Returns
        -------
            residual_error_global_poincare (np.ndarray): Square of the residual error.

        """

        # Obtain the minimum of the permeablity
        perm: np.ndarray = d[pp.PARAMETERS][self.kw]["second_order_tensor"].values
        perm_vals: np.ndarray = perm[0][0].reshape(g.num_cells, 1)
        min_perm: Union[float, int] = np.min(perm_vals)

        # Determine the constant multiplying the residual norm
        const: Union[float, int] = poincare_constant ** 2 / min_perm

        # Retrieve residual norm
        residual_norm: np.ndarray = self.residual_norm_squared(g, g_rot, d, source).copy()

        # Determine the residual error
        residual_error_global_poincare: np.ndarray = const * residual_norm

        return residual_error_global_poincare

    def residual_error_local_poincare(self,
                                      g: pp.Grid,
                                      g_rot: mde.RotatedGrid,
                                      d: dict,
                                      source: Union[Callable[..., np.ndarray], int],
                                      ) -> np.ndarray:
        """Residual error squared using the local Poincare constant.

        Parameters
        ----------
            g (pp.Grid): PorePy grid.
            g_rot (mde.RotatedGrid): Rotated pseudo-grid.
            d (dict): Data dictionary.
            source (Function or 0): External source term applied to the grid. If 0,
                 then it is assumed that no source term should be imposed.

        Returns
        -------
            residual_error_local_poincare (np.ndarray): Square of the residual error.

        """

        # Retrieve the permeability
        perm: np.ndarray = d[pp.PARAMETERS][self.kw]["second_order_tensor"].values
        perm_vals: np.ndarray = perm[0][0]

        # Square of the local Poincare constant
        poincare_const: np.ndarray = (g.cell_diameters() / np.pi) ** 2

        # Constants multiplying the norm
        const: np.ndarray = poincare_const / perm_vals

        # Retrieve the residual norm
        residual_norm: np.ndarray = self.residual_norm_squared(g, g_rot, d, source).copy()

        # Determine the residual error
        residual_error_local_poincare: np.ndarray = const * residual_norm

        return residual_error_local_poincare
