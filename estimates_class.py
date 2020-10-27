#from porepy.estimates import (utility, reconstructions, numerical_integration)
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

import porepy as pp

class ErrorEstimate():
    """
    This is the parent class for the computation of error estimates for
    incompressible flow in mixed-dimensional settings.
    """
    
    def __init__(
        self,
        gb,
        kw="flow",
        sd_operator_name="diffusion",
        p_name="pressure",
        flux_name="flux",
        lam_name="mortar_solution",
        p_rec="direct",
        estimates_kw="estimates"
        ):
    
        self.gb = gb
        self.kw = kw
        self.sd_operator_name =  sd_operator_name
        self.p_name = p_name
        self.flux_name = flux_name
        self.lam_name = lam_name
        self.p_rec = p_rec
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
            + " Pressure reconstruction method: " + self.p_rec + "\n"
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
            d[self.estimates_kw] = { }
        
        # And, loop through all the edges
        for e, d_e in self.gb.edges():
            d_e[self.estimates_kw] = { }
            
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
                subdomain pressures using either a direct reconstruction or 
                inverse of the local pressure gradient. The reconstructed 
                pressure is stored in d['estimates']["rec_p"]. 
        
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
        from porepy.estimates._velocity_reconstruction import (
            compute_full_flux,
            reconstruct_velocity,
            )
        
        # Pressure reconstruction methods
        from porepy.estimates._pressure_reconstruction import(
            reconstruct_pressure,
            )
        
        # Error evaluation methods
        from porepy.estimates._error_evaluation import compute_error_estimates
        
                
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
        
        errors = ["diffusive_error"] # add residual error later
        
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
                transfer(d_e, error)
                    
                
        return None
    
    
    def compute_global_error(self):
        """
        Computes the sum of the local squared errors by looping through 
        all the subdomains and interfaces of the grid bucket.
           
        Raises
        ------
        ValueError
            If the estimates have not been computed
    
        Returns
        -------
        global_error : Scalar
            Global error, i.e., sum of squared errors.
    
        """
        global_error = 0
        errors = ["diffusive_error"] # add residual error later
    
        def add_contribution(data, error_type, global_error):
            if error_type in data[self.estimates_kw]:
                global_error += data[self.estimates_kw][error_type].sum()
            else:
                raise ValueError('Error estimates must be computed first.')
            return global_error
        
        # Add contribution from subdomains
        for g, d in self.gb:
            if g.dim == 0:
                continue
            for error in errors:
                global_error = add_contribution(d, error, global_error)
            
        # Add contribution from interfaces
        for _, d_e in self.gb.edges():
            for error in errors:
                global_error = add_contribution(d_e, error, global_error)
            
        return global_error
    
    
    def compute_local_error(self, g, d, error_type="all"):
        """
        Computes the sum of the local squared error of a subdomain or interface
    
        Parameters
        ----------
        g : PorePy object
            Grid (for subdomains) or mortar grid (for interfaces)
        d : dictionary
            Data dictionary containing the estimates
        error_type: Keyword, optional
            The sum of all errors are computed by default. Other options are
            "diffusive_error", and "residual_error".
    
        Raises
        ------
        ValueError
            If the errors have not been computed.
            If the error_type is not "all", "diffusive_error", or "residual error".
            If there are any inconsistency in the grids dimensions
    
        Returns
        -------
        local_error : Scalar
            Local error, i.e, sum of individual (element) squared errors.
    
        """
        errors = ["diffusive_error"] # add residual error later
        
        # Check if errors have been computed
        def check_error(d, error_type, estimates_kw):
            if error_type not in d[estimates_kw]:
                raise ValueError("Errors must be computed first")
        for error in errors:
            check_error(d, error, self.estimates_kw)
    
        # Check if error type is implemented
        if error_type not in ["all", "diffusive_error"]: # add residual error later
            raise ValueError("Error type not implemented")
    
        # Raise an error if the subdomain is zero-dimensional
        if isinstance(type(g), pp.MortarGrid) and g.dim not in [0, 1, 2]:
            raise ValueError("Invalid dimension, expected 0D, 1D, or 2D.")
        elif g.dim not in [1, 2, 3]:
            raise ValueError("Invalid dimension, expected 1D, 2D, or 3D")
            
        # Summing the errors
        diffusive_error = d[self.estimates_kw]["diffusive_error"].sum()
        #nonconf_error = d[self.estimates_kw]["nonconf_error"].sum()
        all_errors = diffusive_error #+ nonconf_error
        
        # Return the errors
        if error_type == "all":
            local_error = all_errors
        elif error_type == "diffusive_error":
            local_error = diffusive_error
        else:
            #local_error = nonconf_error
            pass
        
        return local_error