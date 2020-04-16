import porepy as pp

import pressure_reconstruction
import flux_reconstruction
import error_evaluation


class PosterioriError:
    "Parent class for computing a posteriori errors for mixed-dimensional equations"

    def __init__(
        self,
        grid_bucket,
        parameter_keyword,
        subdomain_variable,
        nodal_method="k-averaging",
        p_order="1",
    ):
        """
        Class for computing a posteriori error estimates for mixed-dimensional
        flow equations. The errors are included under pp.STATE on the dictionaries
        corresponding to each grid.

        Parameters
        ----------
        grid_bucket : PorePy object
            Grid bucket containing all the grids.
        parameter_keyword : string
            Keyword referring to the problem.
        subdomain_variable : string
            Keyword referring to the subdomain variable.
        nodal_method : string, optional
            Method for obtaining nodal pressures. The default is "k-averaging".
            The other implemented option is "mpfa-inverse".
        p_order : string, optional
            Order of polynomial reconstruction. The default is "1". The other
            implemented option is "1.5", which refers to P1 elements enriched
            with purely parabolic terms.

        Returns
        -------
        None. 

        """
        self.gb = grid_bucket
        self.kw = parameter_keyword
        self.sdv = subdomain_variable

        # Compute the errors on the subdomains by looping through the nodes
        for g, d in self.gb:
            
            if g.dim > 0 :

                # Rotate grid to reference coordinate system
                g_rot = self._rotate_grid(g)
                # Get reconstructed pressure coefficients
                p_coeffs = pressure_reconstruction.subdomain_pressure(
                    g_rot, d, self.kw, self.sdv, nodal_method, p_order
                )
                # Get reconstructed velocity coefficients
                v_coeffs, _ = flux_reconstruction.subdomain_velocity(self.gb, g_rot, g,  d, self.kw)
    
                # Compute different errors
                error_evaluation.diffusive_flux_sd(g_rot, d, p_coeffs, v_coeffs)

        # Compute the errors on the interfaces by looping through the edges
        for e, d in self.gb.edges():
         
            # # Retrieve neighboring grids and mortar grid   
            # g_low, g_high = self.gb.nodes_of_edge(e)
            # mortar_grid = d["mortar_grid"]
            
            # # We assume that the lower dimensional grid coincides geometrically
            # # with the mortar grid. Hence, we rotate the g_low
            # glow_rot = self.rotate(g_low)
            
            # # Get reconstructed interface fluxes
            # lambda_coeffs = flux_reconstruction.interface_flux(glow_rot, mortar_grid, d, self.kw)
            
            
            # # TODO: Calculate errors in the interfaces
            pass

    def _rotate_grid(self, g):
        """
        Rotates grid to account for embedded fractures. 
        
        Note that the pressure and flux reconstruction use the rotated grids, 
        where only the relevant dimensions are taken into account, e.g., a 
        one-dimensional tilded fracture will be represented by a three-dimensional 
        grid, where only the first dimension is used.
        
        Parameters
        ----------
        g : PorePy object
            Original (unrotated) PorePy grid.
    
        Returns
        -------
        g_rot : Porepy object
            Rotated PorePy grid.
            
        """

        # Copy grid to keep original one untouched
        g_rot = g.copy()

        # Rotate grid
        (
            cell_centers,
            face_normals,
            face_centers,
            R,
            dim,
            nodes,
        ) = pp.map_geometry.map_grid(g_rot)

        # Update rotated fields in the relevant dimension
        for dim in range(g.dim):
            g_rot.cell_centers[dim] = cell_centers[dim]
            g_rot.face_normals[dim] = face_normals[dim]
            g_rot.face_centers[dim] = face_centers[dim]
            g_rot.nodes[dim] = nodes[dim]

        # Add the rotation matrix and the effective dimensions to rotated grid
        g_rot.rotation_matrix = R
        g_rot.effective_dim = dim

        return g_rot
