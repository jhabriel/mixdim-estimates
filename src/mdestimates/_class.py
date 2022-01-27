import numpy as np
import porepy as pp
import mdestimates as mde

from typing import Callable, List, Optional, Union, Tuple


class ErrorEstimate:
    """ Main class for the computation of a posteriori error estimates for mD equations."""

    def __init__(
        self,
        gb: pp.GridBucket,
        kw: str = "flow",
        sd_operator_name: str = "diffusion",
        p_name: str = "pressure",
        flux_name: str = "flux",
        lam_name: str = "mortar_flux",
        estimates_kw: str = "estimates",
        p_recon_method: str = "inv_local_gradp",
        source_list: Optional[List[Union[Callable[..., np.ndarray], int]]] = None,
        poincare_list: Optional[List[Union[float, int]]] = None,
    ):
        """
        Computes a posteriori error estimates for the single phase flow in fractured porous
        media. The errors are calculated for each node and edge of the grid bucket. Two main
        errors are estimated: diffusive flux errors on subdomains and interfaces, and
        residual errors errors on subdomains.

        Parameters:
        -----------
            gb (GridBucket): Mixed-dimensional grid bucket. It is assumed that a valid
                pressure solution is stored in d[pp.STATE][self.p_name]. In addition, for
                mixed methods, a valid flux solution is stored in d[pp.STATE][self.flux_name].
            kw (str): Keyword parameter. Default is "flow".
            sd_operator_name (str): Subdomain operator name. Default is "diffusion".
            p_name (str): Pressure name. Default is "pressure".
            flux_name (str): Flux name. Default is "flux".
            lam_name (str): Mortar flux name. Default is "mortar flux".
            estimates_kw (str): Error estimates name. Default is "estimates".
            p_recon_method (str): Pressure reconstruction method. Default is
                "inv_local_gradp". The other valid method is "direct_reconstruction".
            source_list (list): Each item of this list is either a callable function or 0.
                For the first case, the callable function corresponds to the exact
                source term of the given grid. For the second case, it is assumed that no
                external source is imposed. The source list it is assumed to be sorted and
                coinciding with the list generated via [g for _, g in gb]. If source_list is
                not passed, then it is assumed that no external sources are present in the
                problem.
            poincare_list (list): List of Poincare constants for each subdomain. Default is
                1 for each subdomain.
        """

        self.gb: pp.GridBucket = gb
        self.kw: str = kw
        self.sd_operator_name: str = sd_operator_name
        self.p_name: str = p_name
        self.flux_name: str = flux_name
        self.lam_name: str = lam_name
        self.estimates_kw: str = estimates_kw
        self.p_recon_method: str = p_recon_method

        if source_list is None:
            source_list = [0 for _ in gb]
        self.source_list = source_list

        if poincare_list is None:
            poincare_list = [1 for _ in gb]
            self.poincare = "local"
        else:
            self.poincare = "global"
        self.poincare_list = poincare_list

    def __str__(self):
        return "Error estimate object."

    def __repr__(self):
        return (
            "Error estimate object with atributes: "
            + "\n"
            + " Model: "
            + self.kw
            + "\n"
            + " Subdomain operator: "
            + self.sd_operator_name
            + "\n"
            + " Subdomain variable: "
            + self.p_name
            + "\n"
            + " Flux variable: "
            + self.flux_name
            + "\n"
            + " Interface variable: "
            + self.lam_name
            + "\n"
            + " Pressure reconstruction method: "
            + self.p_recon_method
            + "\n"
        )

    def estimate_error(self):
        """
        Main method to estimate the errors in all nodes and edges of the grid bucket.

        Technical note
        --------------

        General algortihm overview:

        [1] Flux-related calculations

            [1.1] Perform reconstruction of the subdomain velocities using RT0
                extension of the normal fluxes and store them in d["estimates"]["rec_u"].

        [2] Pressure-related calculations

            [2.1] Reconstruct the pressure. Perform a P1 reconstruction of the subdomain
                pressures using the inverse of the local pressure gradient. The
                reconstructed pressure is stored in d['estimates']["rec_p"].

        [3] Computation of the error estimates

            [3.1] Compute diffusive errors for the entire grid bucket. The errors (squared)
                are stored element-wise under d[self.estimates_kw]["diffusive_error"].
            [3.2] Compute residual errors for the entire grid bucket. The errors (squared)
                are stored element-wise under d[self.estimates_kw]["residual_error"].

        """

        # Populating data dicitionaries with the key: self.estimates_kw
        self.init_estimates_data_keyword()

        print("Performing velocity reconstruction...", end="")
        vel_rec = mde.VelocityReconstruction(self.gb)
        # 1.1: Compute full flux
        vel_rec.compute_full_flux()
        # 1.2: Reconstruct velocity
        vel_rec.reconstruct_velocity()
        print("\u2713")

        print("Performing pressure reconstruction...", end="")
        p_rec = mde.PressureReconstruction(self.gb)
        # 2.1: Reconstruct pressure
        p_rec.reconstruct_pressure()
        print("\u2713")

        print("Computing upper bounds...", end="")
        # 3.1 Diffusive errors
        diffusive_error = mde.DiffusiveError(self.gb)
        diffusive_error.compute_diffusive_error()
        # 3.2 Residual errors
        residual_error = mde.ResidualError(self.gb)
        residual_error.compute_residual_error()
        print("\u2713")

    def init_estimates_data_keyword(self):
        """
        Initializes the keyword self.estimates_kw in all data dictionaries of the grid bucket.

        """

        # Loop through all the nodes
        for _, d in self.gb:
            d[self.estimates_kw] = {}

        # Loop through all the edges
        for _, d in self.gb.edges():
            d[self.estimates_kw] = {}

    def transfer_error_to_state(self):
        """
        Transfer the results from d[self.estimates_kw] to d[pp.STATE].

        Raises
        ------
            ValueError: If the errors have not been not been computed.

        Note
        -----
            This method is especially useful for exporting the results via pp.Exporter.

        """

        def transfer(data: dict, error_type: str):
            if error_type in data[self.estimates_kw]:
                data[pp.STATE][error_type] = data[self.estimates_kw][error_type].copy()
            else:
                raise ValueError("Estimates must be computed first.")

        # Transfer errors from subdomains to d[pp.STATE]
        for g, d in self.gb:
            if g.dim == 0:
                continue
            for error in ["diffusive_error", "residual_error"]:
                transfer(d, error)

        # Transfer error from interfaces to d[pp.STATE]
        for _, d in self.gb.edges():
            transfer(d, "diffusive_error")

    def get_majorant(self) -> float:
        """
        Computes the majorant of the whole fracture network.

        Returns
        -------
            majorant (float): Majorant of the whole fracture network.

        """

        subdomain_diffusive_squared = 0
        subdomain_residual_squared = 0
        interface_diffusive_squared = 0

        # Errors associated to subdomains
        for g, d in self.gb:
            if g.dim != 0:
                subdomain_diffusive_squared += d[self.estimates_kw][
                    "diffusive_error"
                ].sum()
                subdomain_residual_squared += d[self.estimates_kw][
                    "residual_error"
                ].sum()

        # Errors associated to interfaces
        for _, d in self.gb.edges():
            interface_diffusive_squared += d[self.estimates_kw]["diffusive_error"].sum()

        # Obtaining the majorant
        majorant = np.sqrt(subdomain_diffusive_squared + interface_diffusive_squared) + (
                   np.sqrt(subdomain_residual_squared))

        return majorant

    def get_scaled_majorant(self) -> Union[int, float]:
        """
        Get the permeability-scaled majorant of the whole fracture network.

        Returns
        -------
            scaled_majorant (int or float): Scaled value of the majorant.

        """

        # Determine the largest permeability in the fracture network
        scaling_factors = []
        for g, d in self.gb:
            if g.dim != 0:
                perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
                perm = perm[0][0]
                scaling_factors.append(np.max(perm))
        for _, d in self.gb.edges():
            k_norm = d[pp.PARAMETERS]["flow"]["normal_diffusivity"]
            scaling_factors.append(np.max(k_norm))
        scale_factor = np.max(scaling_factors)

        # Perform scaling
        scaled_majorant = scale_factor ** (-0.5) * self.get_majorant()

        return scaled_majorant

    def get_local_error(self,
                        g: Union[pp.Grid, pp.MortarGrid],
                        d: dict,
                        error_type: str = "all") -> Union[int, float]:
        """
        Computes the sum of the local errors of a subdomain or interface.

        Parameters
        ----------
            g (pp.Grid or pp.MortarGrid): Grid for subdomains or mortar grid for interfaces.
            d (dict): Data dictionary containing the estimates.
            error_type (str): Type of error estimate to be retrieved. Valid entries for
                subdomains are "diffusive_error", "residual_error", or "all", that is,
                the sum of diffusive and residual errors. Valid entries for interfaces are
                "diffusive_error" or "all". Default is "all".
        Raises
        ------
            ValueError
                (*) If there is any inconsistency in the grids dimensions.
                (*) If the errors have not been computed.
                (*) If any error besides "diffusive_error","residual_error", or "all" is
                    requested.

        Returns
        -------
            error (int or float): Local error, i.e., square root of the sum of individual
                squared errors.

        """

        # Boolean variable to check if g is pp.MortarGrid or pp.Grid
        is_mortar: bool = issubclass(type(g), pp.MortarGrid)

        # Raise an error if an invalid error type is requested
        if is_mortar:
            if error_type not in ["diffusive_error", "all"]:
                raise ValueError("Invalid error type for interfaces. See documentation.")
        else:
            if error_type not in ["diffusive_error", "residual_error", "all"]:
                raise ValueError("Invalid error type for subdmains. See documentation.")

        # Raise an error if an invalid grid dimension is passed
        if is_mortar:
            if g.dim not in [0, 1, 2]:
                raise ValueError("Invalid dimension for mortar grid. Expected 0, 1, or 2.")
        else:
            if g.dim not in [1, 2, 3]:
                raise ValueError("Invalid dimension for grid. Expected 1, 2, or 3.")

        # Raise an error if the errors have not been computed
        if is_mortar:
            if "diffusive_error" not in d[self.estimates_kw]:
                raise ValueError("Interface errors must be computed first.")
        else:
            if "diffusive_error" and "residual_error" not in d[self.estimates_kw]:
                raise ValueError("Subdomain errors must be computed first.")

        # Retrieve the requested error
        if is_mortar:
            error = d[self.estimates_kw]["diffusive_error"].sum() ** 0.5
        else:
            if error_type == "diffusive_error":
                error = d[self.estimates_kw]["diffusive_error"].sum() ** 0.5
            elif error_type == "residual_error":
                error = d[self.estimates_kw]["residual_error"].sum() ** 0.5
            else:
                error = (d[self.estimates_kw]["diffusive_error"].sum() +
                         d[self.estimates_kw]["residual_error"].sum()) ** 0.5

        return error

    def get_scaled_local_errors(self,
                                g: Union[pp.Grid, pp.MortarGrid],
                                d: dict,
                                error_type: str = "all") -> Union[int, float]:
        """
        Computes the sum of the scaled local errors of a subdomain or interface.

        Parameters
        ----------
            g (pp.Grid or pp.MortarGrid): Grid for subdomains or mortar grid for interfaces.
            d (dict): Data dictionary containing the estimates.
            error_type (str): Type of error estimate to be retrieved. Valid entries for
                subdomains are "diffusive_error", "residual_error", or "all", that is,
                the sum of diffusive and residual errors. Valid entries for interfaces are
                "diffusive_error" or "all". Default is "all".
        Raises
        ------
            ValueError
                (*) If there is any inconsistency in the grids dimensions.
                (*) If the errors have not been computed.
                (*) If any error besides "diffusive_error","residual_error", or "all" is
                    requested.

        Returns
        -------
            scaled_error (int or float): Scaled local error, i.e., square root of the sum of
                individual squared errors.

        """

        # Boolean variable to check if g is pp.MortarGrid or pp.Grid
        is_mortar: bool = issubclass(type(g), pp.MortarGrid)

        # Raise an error if an invalid error type is requested
        if is_mortar:
            if error_type not in ["diffusive_error", "all"]:
                raise ValueError("Invalid error type for interfaces. See documentation.")
        else:
            if error_type not in ["diffusive_error", "residual_error", "all"]:
                raise ValueError("Invalid error type for subdmains. See documentation.")

        # Raise an error if an invalid grid dimension is passed
        if is_mortar:
            if g.dim not in [0, 1, 2]:
                raise ValueError("Invalid dimension for mortar grid. Expected 0, 1, or 2.")
        else:
            if g.dim not in [1, 2, 3]:
                raise ValueError("Invalid dimension for grid. Expected 1, 2, or 3.")

        # Raise an error if the errors have not been computed
        if is_mortar:
            if "diffusive_error" not in d[self.estimates_kw]:
                raise ValueError("Interface errors must be computed first.")
        else:
            if "diffusive_error" and "residual_error" not in d[self.estimates_kw]:
                raise ValueError("Subdomain errors must be computed first.")

        # Retrieving the requested scaled error
        if is_mortar:
            k_perp = d[pp.PARAMETERS][self.kw]["normal_diffusivity"]
            scaled_error = np.sum(
                (1 / k_perp) * d[self.estimates_kw]["diffusive_error"]
            ) ** 0.5
        else:
            if error_type == "diffusive_error":
                perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
                perm = perm[0][0]
                scaled_error = np.sum(
                    (1 / perm) * d[self.estimates_kw]["diffusive_error"]
                ) ** 0.5
            elif error_type == "residual_error":
                scaled_error = np.sum(d[self.estimates_kw]["residual_error"]) ** 0.5
            else:
                perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
                perm = perm[0][0]
                scaled_error = (np.sum(
                    (1 / perm) * d[self.estimates_kw]["diffusive_error"])
                                + np.sum(d[self.estimates_kw]["residual_error"])) ** 0.5

        return scaled_error

    def print_summary(self, scaled: bool = False):
        """
        Print summary of errors. If scaled is True, the scaled version of the errors is used.

        Parameters
        ----------
            scaled (bool): Wheter the scaled version of the errors will be printed or not. The
                default is False.

        """

        if scaled:
            self._print_summary_scaled()
        else:
            self._print_summary_original()

    def _print_summary_original(self):
        """
        Prints summary of the errors.

        """

        # Get hold of max and min dims
        dim_max = self.gb.dim_max()
        dim_min = self.gb.dim_min()

        # Obtain dimensions of subdomains and interfaces
        dims = np.arange(start=dim_min, stop=dim_max + 1)

        subdomain_dims = dims[::-1]
        if dim_min == 0:
            subdomain_dims = subdomain_dims[: subdomain_dims.size - 1]

        interface_dims = dims[::-1]  # sort
        interface_dims = interface_dims[1::]  # ignore first element

        # Get scaled majorant and print it
        majorant = self.get_majorant()
        print("Majorant:", majorant)

        # Print summary of subdomain errors
        for dim in subdomain_dims:
            g_list = self.gb.grids_of_dimension(dim)
            error = 0
            for g in g_list:
                d = self.gb.node_props(g)
                error += self.get_local_error(g, d)
            print(f"{dim}D Subdomain error: {error}")

        # Print summary of interface errors
        for dim in interface_dims:
            error = 0
            for _, d in self.gb.edges():
                mg = d["mortar_grid"]
                if mg.dim == dim:
                    error += self.get_local_error(mg, d)
            print(f"{dim}D Interface error: {error}")

    def _print_summary_scaled(self):
        """
        Prints summary of the scaled errors.

        """

        # Get hold of max and min dims
        dim_max = self.gb.dim_max()
        dim_min = self.gb.dim_min()

        # Obtain dimensions of subdomains and interfaces
        dims = np.arange(start=dim_min, stop=dim_max + 1)

        subdomain_dims = dims[::-1]
        if dim_min == 0:
            subdomain_dims = subdomain_dims[: subdomain_dims.size - 1]

        interface_dims = dims[::-1]  # sort
        interface_dims = interface_dims[1::]  # ignore first element

        # Get scaled majorant and print it
        scaled_majorant = self.get_scaled_majorant()
        print("Scaled majorant:", scaled_majorant)

        # Print summary of subdomain errors
        for dim in subdomain_dims:
            g_list = self.gb.grids_of_dimension(dim)
            error = 0
            for g in g_list:
                d = self.gb.node_props(g)
                error += self.get_scaled_local_errors(g, d)
            print(f"{dim}D Subdomain scaled error: {error}")

        # Print summary of interface errors
        for dim in interface_dims:
            error = 0
            for _, d in self.gb.edges():
                mg = d["mortar_grid"]
                if mg.dim == dim:
                    error += self.get_scaled_local_errors(mg, d)
            print(f"{dim}D Interface scaled error: {error}")
