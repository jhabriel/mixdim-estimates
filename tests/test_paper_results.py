import numpy as np
import porepy as pp
import pytest

from paper_examples.ex51_validation2D.model_local import model_local


def test_2d_validation_rt0_p0():
    """Tests the results of the 2d validation"""

    def make_constrained_mesh(h=0.1):
        """
        Creates unstructured mesh for a given target mesh size for the case of a
        single vertical fracture embedded in the domain

        Parameters
        ----------
        h : float, optional
            Target mesh size. The default is 0.1.

        Returns
        -------
        gb : PorePy Object
            Porepy grid bucket object.

        """

        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network_2d = pp.fracture_importer.network_2d_from_csv("network.csv", domain=domain)

        # Target lengths
        target_h_bound = h
        target_h_fract = h
        mesh_args = {
            "mesh_size_bound": target_h_bound,
            "mesh_size_frac": target_h_fract,
        }
        # Construct grid bucket
        gb = network_2d.mesh(mesh_args, constraints=[1, 2])

        return gb

    # Create grid bucket
    gb = make_constrained_mesh(h=0.05)

    # Retrieve dictionary from model
    d = model_local(gb, "RT0")

    # Obtained majorant
    majorant = d["majorant_pressure"]
    i_eff_p = d["efficiency_pressure"]
    i_eff_u = d["efficiency_velocity"]
    i_eff_pu = d["efficiency_combined"]

    # Known majorant
    known_majorant = 4.36e-02
    known_i_eff_p = 1.08
    known_i_eff_u = 3.04
    known_i_eff_pu = 1.59

    np.testing.assert_almost_equal(known_majorant, majorant, decimal=4)
    np.testing.assert_almost_equal(known_i_eff_p, i_eff_p, decimal=2)
    np.testing.assert_almost_equal(known_i_eff_u, i_eff_u, decimal=2)
    np.testing.assert_almost_equal(known_i_eff_pu, i_eff_pu, decimal=2)
