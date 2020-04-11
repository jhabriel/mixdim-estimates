import porepy as pp

import pressure_reconstruction
import flux_reconstruction
import error_evaluation

class PosterioriError():
    "Parent class for computing A posteriori errors for mixed-dimensional equations"
    
    def __init__(self, grid_bucket, parameter_keyword, subdomain_variable, nodal_method="k-averaging", p_order="1"):
        """
        Class for computing a posteriori error estimates of mixed-dimensional
        flow equations

        Parameters
        ----------
        grid_bucket : PorePy object
            Grid bucket containing all the grids.
        parameter_keyword : string
            Keyword referring to the problem.
        subdomain_variable : string
            Keyword referring to the subdomain variable.
        nodal_method : TYPE, optional
            DESCRIPTION. The default is "k-averaging".
        p_order : TYPE, optional
            DESCRIPTION. The default is "1".

        Returns
        -------
        None.

        """
        self.gb = grid_bucket
        self.kw = parameter_keyword
        self.sdv = subdomain_variable
        
        # Loop through the nodes
        for g, d in self.gb:          
            
            # Rotate grid to reference coordinate system
            g_rot = self._rotate_grid(g)
            # Get reconstructed pressure coefficients
            p_coeffs = pressure_reconstruction.subdomain_pressure(g_rot, d, self.kw, self.sdv, nodal_method, p_order)
            # Get reconstructed velocity coefficients
            v_coeffs, _ = flux_reconstruction.subdomain_velocity(g_rot, d, self.kw)               
            
            # Compute different errors
            error_evaluation.diffusive_flux_sd(g_rot, d, p_coeffs, v_coeffs);
    

        # Loop through the edges
        for e, d in self.gb.edges():
            # TODO: Calculate errors in the interfaces
            pass
       

    def _rotate_grid(self, g):
        """
        Rotates grid to account for embedded fractures
        
        Parameters
        ----------
        g (PorePy object): Original grid
    
        Returns
        -------
        g_rot (Porepy object): Rotated grid
        """
        
        # Copy grid to keep original one untouched
        g_rot = g.copy()  
        
        # Rotate grid
        cell_centers, face_normals, face_centers, R, dim, nodes = pp.map_geometry.map_grid(g_rot)
        
        # Update rotated fields in the relevant dimension
        for dim in range(g.dim):
            g_rot.cell_centers[dim] = cell_centers[dim]
            g_rot.face_normals[dim] = face_normals[dim]
            g_rot.face_centers[dim] = face_centers[dim]
            g_rot.nodes[dim] = nodes[dim]
        
        return g_rot
    
    
        
        
