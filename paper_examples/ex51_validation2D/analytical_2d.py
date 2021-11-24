import porepy as pp
import numpy as np
import sympy as sym
import quadpy as qp
import mdestimates.estimates_utils as utils

from typing import Tuple


class ExactSolution2D:

    def __init__(self, gb: pp.GridBucket):

        # Grid bucket
        self.gb: pp.GridBucket = gb

        # 2D grid and its dictionary
        self.g2d: pp.Grid = self.gb.grids_of_dimension(2)[0]
        self.d2d: dict = self.gb.node_props(self.g2d)

        # 1D grid and its dictionary
        self.g1d: pp.Grid = self.gb.grids_of_dimension(1)[0]
        self.d1d: dict = self.gb.node_props(self.g1d)
        self.g1d_rot = utils.rotate_embedded_grid(self.g1d)

        # Mortar grid and its dictionary
        self.e: Tuple[pp.Grid, pp.Grid] = (self.g2d, self.g1d)
        self.de: dict = self.gb.edge_props((self.g2d, self.g1d))
        self.mg: pp.MortarGrid = self.de["mortar_grid"]

        # Get list of cell indices
        bottom_cells = self.g2d.cell_centers[1] < 0.25
        middle_cells = (self.g2d.cell_centers[1] >= 0.25) & (self.g2d.cell_centers[1] <= 0.75)
        top_cells = self.g2d.cell_centers[1] > 0.75
        self.cell_idx: list = [bottom_cells, middle_cells, top_cells]

        # Get list of boundary indices
        bottoms_bc = self.g2d.face_centers[1] < 0.25
        middle_bc = (self.g2d.face_centers[1] >= 0.25) & (self.g2d.face_centers[1] <= 0.75)
        top_bc = self.g2d.face_centers[1] > 0.75
        self.bc_idx: list = [bottoms_bc, middle_bc, top_bc]

        # Save as private attributes some useful expressions
        x, y = sym.symbols("x y")
        self._alpha = 1.5
        self._dist_top = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5
        self._dist_mid = ((x - 0.5) ** 2) ** 0.5
        self._dist_bot = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
        self._bubble = (y - 0.25) ** 2 * (y - 0.75) ** 2

    def __repr__(self) -> str:
        return "Exact solution  object for 2D problem"

    def p2d(self, which="cc"):
        """
        Computation of the exact pressure in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
            expressions, and "fun" for the lambda functions depending on (x, y).

        Returns
        -------
        Appropiate object according to which.

        """

        cc = self.g2d.cell_centers
        x, y = sym.symbols("x y")

        p2_top = self._dist_top ** (1 + self._alpha)
        p2_mid = self._dist_mid ** (1 + self._alpha) + self._bubble * self._dist_mid
        p2_bot = self._dist_bot ** (1 + self._alpha)

        p2_sym = [p2_bot, p2_mid, p2_top]
        p2_fun = [sym.lambdify((x, y), p, "numpy") for p in p2_sym]
        p2_cc = np.zeros(self.g2d.num_cells)
        for (p, idx) in zip(p2_fun, self.cell_idx):
            p2_cc += p(cc[0], cc[1]) * idx

        if which == "sym":
            return p2_sym
        elif which == "fun":
            return p2_fun
        else:
            return p2_cc

    def gradp2d(self, which="cc"):
        """
        Computation of the pressure gradients in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
            expressions, and "fun" for the lambda functions depending on (x, y).

        Returns
        -------
        Appropiate object according to which.

        """
        x, y = sym.symbols("x y")
        cc = self.g2d.cell_centers

        gradp2_sym = [[sym.diff(p, x), sym.diff(p, y)] for p in self.p2d(which="sym")]

        gradp2_fun = [
            [
                sym.lambdify((x, y), gradp[0], "numpy"),
                sym.lambdify((x, y), gradp[1], "numpy")
            ]
            for gradp in gradp2_sym
        ]

        gradp2x_cc = np.zeros(self.g2d.num_cells)
        gradp2y_cc = np.zeros(self.g2d.num_cells)
        gradp2z_cc = np.zeros(self.g2d.num_cells)
        for (u, idx) in zip(gradp2_fun, self.cell_idx):
            gradp2x_cc += u[0](cc[0], cc[1]) * idx
            gradp2y_cc += u[1](cc[0], cc[1]) * idx
        gradp2_cc = np.array([gradp2x_cc, gradp2y_cc, gradp2z_cc])

        if which == "sym":
            return gradp2_sym
        elif which == "fun":
            return gradp2_fun
        else:
            return gradp2_cc

    def u2d(self, which="cc"):
        """
        Computation of the exact fluxes in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
            expressions, and "fun" for the lambda functions depending on (x, y).

        Returns
        -------
        Appropiate object according to which.

        """

        x, y = sym.symbols("x y")
        cc = self.g2d.cell_centers

        u2_sym = [[-sym.diff(p, x), -sym.diff(p, y)] for p in self.p2d(which="sym")]

        u2_fun = [
            [
                sym.lambdify((x, y), u[0], "numpy"),
                sym.lambdify((x, y), u[1], "numpy")
            ]
            for u in u2_sym
        ]

        ux_cc = np.zeros(self.g2d.num_cells)
        uy_cc = np.zeros(self.g2d.num_cells)
        uz_cc = np.zeros(self.g2d.num_cells)
        for (u, idx) in zip(u2_fun, self.cell_idx):
            ux_cc += u[0](cc[0], cc[1]) * idx
            uy_cc += u[1](cc[0], cc[1]) * idx
        u2_cc = np.array([ux_cc, uy_cc, uz_cc])

        if which == "sym":
            return u2_sym
        elif which == "fun":
            return u2_fun
        else:
            return u2_cc

    def f2d(self, which="cc"):
        """
        Computation of the exact sources in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
            expressions, and "fun" for the lambda functions depending on (x, y).

        Returns
        -------
        Appropiate object according to which.

        """
        x, y = sym.symbols("x y")
        cc = self.g2d.cell_centers

        f2_sym = [sym.diff(u[0], x) + sym.diff(u[1], y) for u in self.u2d(which="sym")]

        f2_fun = [sym.lambdify((x, y), f, "numpy") for f in f2_sym]

        f2_cc = np.zeros(self.g2d.num_cells)
        for (f, idx) in zip(f2_fun, self.cell_idx):
            f2_cc += f(cc[0], cc[1]) * idx

        if which == "sym":
            return f2_sym
        elif which == "fun":
            return f2_fun
        else:
            return f2_cc

    def lmbda(self, which="cc"):
        """
        Computation of the exact mortar flux

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
            expressions, and "fun" for the lambda functions depending on (y).

        Returns
        -------
        Appropiate object according to which.

        """
        y = sym.symbols("y")
        cc = self.mg.cell_centers

        lmbda_sym = self._bubble

        lmbda_fun = sym.lambdify(y, lmbda_sym, "numpy")

        lmbda_cc = lmbda_fun(cc[1])

        if which == "sym":
            return lmbda_sym
        elif which == "fun":
            return lmbda_fun
        else:
            return lmbda_cc

    def p1d(self, which="cc"):
        """
        Computation of the exact fracture pressure

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
           expressions, and "fun" for the lambda functions depending on (y).

        Returns
        -------
        Appropiate object according to which.

        """
        y = sym.symbols("y")
        cc = self.g1d.cell_centers

        # Symbolic expression
        p1_sym = -1 * self._bubble

        # Function
        p1_fun = sym.lambdify(y, p1_sym, "numpy")

        # Cell center pressures (in real coordinates)
        p1_cc = p1_fun(cc[1])

        # Cell center pressures (in rotated coordinates)
        p1 = np.array([np.zeros(self.g1d.num_cells), p1_cc, np.zeros(self.g1d.num_cells)])
        p1_rot_cc = np.dot(self.g1d_rot.rotation_matrix, p1)[self.g1d_rot.dim_bool][0]

        if which == "sym":
            return p1_sym
        elif which == "fun":
            return p1_fun
        elif which == "cc_rot":
            return p1_rot_cc
        else:
            return p1_cc

    def gradp1d(self, which="cc"):
        """
        Computation of the exact fracture pressure gradient

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
           expressions, and "fun" for the lambda functions depending on (y).

        Returns
        -------
        Appropiate object according to which.

        """
        y = sym.symbols("y")
        cc = self.g1d.cell_centers

        gradp1_sym = sym.diff(self.p1d(which="sym"), y)

        gradp1_fun = sym.lambdify(y, gradp1_sym, "numpy")

        gradp1_cc = gradp1_fun(cc[1])

        if which == "sym":
            return gradp1_sym
        elif which == "fun":
            return gradp1_fun
        else:
            return gradp1_cc

    def u1d(self, which="cc"):
        """
        Computation of the exact fracture velocity

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
           expressions, and "fun" for the lambda functions depending on (y).

        Returns
        -------
        Appropiate object according to which.

        """
        y = sym.symbols("y")
        cc = self.g1d.cell_centers

        u1_sym = -sym.diff(self.p1d(which="sym"), y)

        u1_fun = sym.lambdify(y, u1_sym, "numpy")

        u1_cc = u1_fun(cc[1])

        if which == "sym":
            return u1_sym
        elif which == "fun":
            return u1_fun
        else:
            return u1_cc

    def f1d(self, which="cc"):
        """
        Computation of the exact fracture source

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the symbolic
           expressions, and "fun" for the lambda functions depending on (y).

        Returns
        -------
        Appropiate object according to which.

        """
        y = sym.symbols("y")
        cc = self.g1d.cell_centers

        jump_lmbda_sym = 2 * self.lmbda(which="sym")
        div_u1_sym = sym.diff(self.u1d(which="sym"), y)
        f1_sym = div_u1_sym - jump_lmbda_sym

        f1_fun = sym.lambdify(y, f1_sym, "numpy")

        f1_cc = f1_fun(cc[1])

        if which == "sym":
            return f1_sym
        elif which == "fun":
            return f1_fun
        else:
            return f1_cc

    def dir_bc_values(self):
        """
        Computation of the exact fracture source

        Returns
        -------
        NumPy array containing the pressure boundary values

        """

        # Get face-center coordinates
        fc = self.g2d.face_centers

        # Initialize boundary values array
        bc_values = np.zeros(self.g2d.num_faces)

        # Evaluate exact pressure at external boundary faces at each region
        for (p, idx) in zip(self.p2d("fun"), self.bc_idx):
            bc_values[idx] = p(fc[0][idx], fc[1][idx])

        return bc_values

    def integrate_f2d(self):

        # Declare integration method and get hold of elements in QuadPy format
        int_method = qp.t2.get_good_scheme(10)
        elements = utils.get_quadpy_elements(self.g2d, self.g2d)

        integral = np.zeros(self.g2d.num_cells)
        for (f, idx) in zip(self.f2d("fun"), self.cell_idx):
            # Declare integrand
            def integrand(x):
                return f(x[0], x[1]) * np.ones_like(x[0])
            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    def integrate_f1d(self):

        method = qp.c1.newton_cotes_closed(5)
        g_rot = utils.rotate_embedded_grid(self.g1d)
        elements = utils.get_quadpy_elements(self.g1d, g_rot)
        elements *= -1  # we have to use the real y coordinates here

        def integrand(y):
            return self.f1d("fun")(y)

        integral = method.integrate(integrand, elements)

        return integral
