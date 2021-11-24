# from porepy.estimates import (utility, reconstructions, numerical_integration)
from typing import (
    Any,
    Tuple,
    Dict,
    Generator,
    List,
    Iterable,
    Callable,
    Union,
    TypeVar,
    Generic,
)

import numpy as np
import porepy as pp


class ErrorEstimate():
    """
    Parent class for computation of a posteriori error estimates for solutions
    of the incompressible flow in mixed-dimensional geometries.
    """

    def __init__(
            self,
            gb,
            kw="flow",
            sd_operator_name="diffusion",
            p_name="pressure",
            flux_name="flux",
            lam_name="mortar_flux",
            estimates_kw="estimates"
    ):

        self.gb = gb
        self.kw = kw
        self.sd_operator_name = sd_operator_name
        self.p_name = p_name
        self.flux_name = flux_name
        self.lam_name = lam_name
        self.estimates_kw = estimates_kw

    def __str__(self):
        return "Error estimate object"

    def __repr__(self):
        return (
                "Error estimate object with atributes: " + "\n"
                + " Model: " + self.kw + "\n"
                + " Subdomain operator: " + self.sd_operator_name + "\n"
                + " Subdomain variable: " + self.p_name + "\n"
                + " Flux variable: " + self.flux_name + "\n"
                + " Interface variable: " + self.lam_name + "\n"
        )

    def _init_estimates_data_keyword(self):
        """
        Private method that initializes the keyword [self.estimates_kw] inside
        the data dictionary for all nodes and edges of the entire grid bucket.
    
        Returns
        -------
        None.
    
        """
        # Loop through all the nodes
        for g, d in self.gb:
            d[self.estimates_kw] = {}

        # And, loop through all the edges
        for e, d_e in self.gb.edges():
            d_e[self.estimates_kw] = {}

        return None

    def estimate_error(self):
        """
        Main method to estimate the a posteriori errors. This method relies
        on other private methods (see below). 

                            - GENERAL ALGORITHM OVERVIEW - 
            
        [1] Flux-related calculations
            
            # 1.1 Compute full flux for each node of the grid bucket, and store 
                them in d["estimates"]["full_flux"]
            
            # 1.2 Perform reconstruction of the subdomain velocities using RT0 
                extension of the normal fluxes and store them in 
                d["estimates"]["rec_u"]
                
        [2] Pressure-related calculations 
        
            # 2.1 Reconstruct the pressure. Perform a P1 reconstruction of the
                subdomain pressures using the inverse of the local pressure 
                gradient. The reconstructed pressure is stored 
                in d['estimates']["rec_p"]. 
        
        [3] Computation of the upper bounds and norms
        
            # 3.1 Compute errors for the entire grid bucket. The errors 
            (squared) are stored element-wise under 
            d[self.estimates_kw]["diffusive_error"] and 
            d[self.estimates_kw]["residual_error"], respectivley.

        Returns
        -------
        None.

        """

        # Velocity reconstruction methods
        from mdestimates._velocity_reconstruction import (
            compute_full_flux,
            reconstruct_velocity,
        )

        # Pressure reconstruction methods
        from mdestimates._pressure_reconstruction import (
            reconstruct_pressure,
        )

        # Error evaluation methods
        from mdestimates._error_evaluation import compute_error_estimates

        # Populating data dicitionaries with self.estimates_kw
        self._init_estimates_data_keyword()

        print("Performing velocity reconstruction...", end="")
        # 1.1: Compute full flux
        compute_full_flux(self)

        # 1.2: Reconstruct velocity        
        reconstruct_velocity(self)
        print("\u2713")

        print("Performing pressure reconstruction...", end="")
        # 2.1: Reconstruct pressure
        reconstruct_pressure(self)
        print("\u2713")

        print("Computing upper bounds...", end="")
        # 3.1 Evaluate norms and compute upper bounds
        compute_error_estimates(self)
        print("\u2713")

    def transfer_error_to_state(self):
        """
        Transfers the results from d[self.estimates_kw] to d[pp.STATE] for each 
        node and edge of the grid bucket. This method is especially useful
        for exporting the results to Paraview via pp.Exporter.
    
        Raises
        ------
        ValueError
            If the errors have not been not been computed.
    
        Returns
        -------
        None.
    
        """

        errors = ["diffusive_error", "residual_error"]

        def transfer(d, error_type):
            if error_type in d[self.estimates_kw]:
                d[pp.STATE][error_type] = d[self.estimates_kw][error_type].copy()
            else:
                raise ValueError("Estimates must be computed first")

        # Transfer errors from subdomains
        for g, d in self.gb:
            if g.dim == 0:
                continue
            for error in errors:
                transfer(d, error)

        # Transfer error from interfaces     
        for _, d_e in self.gb.edges():
            for error in errors:
                if error == "diffusive_error":
                    transfer(d_e, error)

        return None

    def get_majorant(self):
        """
            Computes the majorant for the whole fracture network.

        Returns
        -------
        majorant : Scalar
            Global error estimate.

        """

        subdomain_diffusive_squared = 0
        mortar_diffusive_squared = 0

        # Errors associated to subdomains
        for g, d in self.gb:
            if g.dim != 0:
                subdomain_diffusive_squared += d[self.estimates_kw]["diffusive_error"].sum()

        # Errors associated to interfaces
        for _, d in self.gb.edges():
            mortar_diffusive_squared += d[self.estimates_kw]["diffusive_error"].sum()

        # Obtaining the majorant
        majorant = np.sqrt(subdomain_diffusive_squared + mortar_diffusive_squared)

        return majorant

    def get_scaled_majorant(self):
        """
        Get the permeability-scaled majorant for the whole fracture network.

        Returns
        -------
        scaled_majorant : Scalar
            Scaled value of the majorant.

        """

        # Determine the highest permeability in the fracture network
        scaling_factors = []
        for g, d in self.gb:
            if g.dim != 0:
                perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
                perm = perm[0][0]
                scaling_factors.append(np.max(perm))
        for e, d in self.gb.edges():
            k_norm = d[pp.PARAMETERS]["flow"]['normal_diffusivity']
            scaling_factors.append(np.max(k_norm))
        scale_factor = np.max(scaling_factors)

        # Perform scaling
        scaled_majorant = scale_factor ** (-0.5) * self.get_majorant()

        return scaled_majorant

    def get_local_errors(self, g, d):
        """
        Computes the sum of the scaled local errors of a subdomain or interface
    
        Parameters
        ----------
        g : PorePy object
            Grid (for subdomains) or mortar grid (for interfaces)
        d : dictionary
            Data dictionary containing the estimates

        Raises
        ------
        ValueError
            If the errors have not been computed.
            If there are any inconsistency in the grids dimensions
    
        Returns
        -------
        local_error : Scalar
            Local error, i.e, sum of individual (element) squared errors.
    
        """

        # Boolean variable to check if it is a mortar grid or not
        is_mortar = issubclass(type(g), pp.MortarGrid)

        # Check if the errors are stored in the data dictionary
        if "diffusive_error" not in d[self.estimates_kw]:
            raise ValueError("Errors must be computed first")

        # Check dimensions of subdomain and mortar grids
        if is_mortar and g.dim not in [0, 1, 2]:
            raise ValueError("Invalid dimension, expected 0D, 1D, or 2D.")
        elif not is_mortar and g.dim not in [1, 2, 3]:
            raise ValueError("Invalid dimension, expected 1D, 2D, or 3D")

        # Summing the errors
        diffusive_error = d[self.estimates_kw]["diffusive_error"].sum()

        return np.sqrt(diffusive_error)

    def get_scaled_local_errors(self, g, d):
        """
        Computes the sum of the scaled local errors of a subdomain or interface
    
        Parameters
        ----------
        g : PorePy object
            Grid (for subdomains) or mortar grid (for interfaces)
        d : dictionary
            Data dictionary containing the estimates

        Raises
        ------
        ValueError
            If the errors have not been computed.
            If there are any inconsistency in the grids dimensions
    
        Returns
        -------
        local_error : Scalar
            Local error, i.e, sum of individual (element) squared errors.
        """

        # Boolean variable to check if it is a Mortar grid or not
        is_mortar = issubclass(type(g), pp.MortarGrid)

        # Check if the errors are stored in the data dictionary
        if "diffusive_error" not in d[self.estimates_kw]:
            raise ValueError("Errors must be computed first")

        # Check dimensions of subdomain and mortar grids
        if is_mortar and g.dim not in [0, 1, 2]:
            raise ValueError("Invalid dimension, expected 0D, 1D, or 2D.")
        elif not is_mortar and g.dim not in [1, 2, 3]:
            raise ValueError("Invalid dimension, expected 1D, 2D, or 3D")

        # Summing the errors
        if is_mortar:
            k_perp = d[pp.PARAMETERS][self.kw]['normal_diffusivity']
            diffusive_error = np.sum((1 / k_perp) * d[self.estimates_kw]["diffusive_error"])
        else:
            perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
            perm = perm[0][0]
            diffusive_error = np.sum((1 / perm) * d[self.estimates_kw]["diffusive_error"])

        return np.sqrt(diffusive_error)

    def print_summary(self, scaled=True):
        """
        Wrapper for printing a summary of the global and local errors for the
        whole fracture network classified by topological dimension. By default,
        the scaled version of the errors are printed.
        
        Parameters
        ----------
        scaled: Bool
            Wheter the scaled version of the errors will be printed or not. The
            default is True.
        
        Returns
        -------
        None.

        """

        if scaled:
            self._print_summary_scaled()
        else:
            self._print_summary_original()

    def _print_summary_original(self):
        """
        Prints summary of the global and local errors

        Returns
        -------
        None.

        """

        # Get hold of max and min dims
        dim_max = self.gb.dim_max()
        dim_min = self.gb.dim_min()

        # Obtain dimensions of subdomains and interfaces
        dims = np.arange(start=dim_min, stop=dim_max + 1)

        subdomain_dims = dims[::-1]
        if dim_min == 0:
            subdomain_dims = subdomain_dims[:subdomain_dims.size - 1]

        interface_dims = dims[::-1]  # sort
        interface_dims = interface_dims[1::]  # ignore first element

        # Get scaled majorant and print it
        majorant = self.get_majorant
        print("Majorant:", majorant)

        # Print summary of subdomain errors
        for dim in subdomain_dims:
            g_list = self.gb.grids_of_dimension(dim)
            error = 0
            for g in g_list:
                d = self.gb.node_props(g)
                error += self.get_local_errors(g, d)
            print(f'{dim}D Subdomain error: {error}')

        # Print summary of interface errors
        for dim in interface_dims:
            error = 0
            for _, d in self.gb.edges():
                mg = d['mortar_grid']
                if mg.dim == dim:
                    error += self.get_local_errors(mg, d)
            print(f'{dim}D Interface error: {error}')

    def _print_summary_scaled(self):
        """
        Prints summary of scaled global and local errors

        Returns
        -------
        None.

        """

        # Get hold of max and min dims
        dim_max = self.gb.dim_max()
        dim_min = self.gb.dim_min()

        # Obtain dimensions of subdomains and interfaces
        dims = np.arange(start=dim_min, stop=dim_max + 1)

        subdomain_dims = dims[::-1]
        if dim_min == 0:
            subdomain_dims = subdomain_dims[:subdomain_dims.size - 1]

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
            print(f'{dim}D Subdomain scaled error: {error}')

        # Print summary of interface errors
        for dim in interface_dims:
            error = 0
            for _, d in self.gb.edges():
                mg = d['mortar_grid']
                if mg.dim == dim:
                    error += self.get_scaled_local_errors(mg, d)
            print(f'{dim}D Interface scaled error: {error}')