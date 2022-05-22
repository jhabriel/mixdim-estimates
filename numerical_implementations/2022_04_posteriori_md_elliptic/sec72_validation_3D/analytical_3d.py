import mdestimates
import porepy as pp
import numpy as np
import sympy as sym
import quadpy as qp
import mdestimates.estimates_utils as utils

from typing import List, Tuple


class ExactSolution3D:
    def __init__(self, gb: pp.GridBucket):

        # Grid bucket
        self.gb: pp.GridBucket = gb

        # 3D grid and its dictionary
        self.g3d: pp.Grid = self.gb.grids_of_dimension(3)[0]
        self.d3d: dict = self.gb.node_props(self.g3d)

        # 2D grid, its dictionary and the corresponding rotated grid
        self.g2d: pp.Grid = self.gb.grids_of_dimension(2)[0]
        self.d2d: dict = self.gb.node_props(self.g2d)
        self.g2d_rot = mdestimates.RotatedGrid(self.g2d)

        # Edge, its dictionary, and the mortar grid
        self.e: Tuple[pp.Grid, pp.Grid] = (self.g3d, self.g2d)
        self.de: dict = self.gb.edge_props((self.g3d, self.g2d))
        self.mg: pp.MortarGrid = self.de["mortar_grid"]

        # Get list of cell indices
        cc = self.g3d.cell_centers
        bottom_front_cc = (cc[1] < 0.25) & (cc[2] < 0.25)
        bottom_middle_cc = (cc[1] < 0.25) & (cc[2] > 0.25) & (cc[2] < 0.75)
        bottom_back_cc = (cc[1] < 0.25) & (cc[2] > 0.75)
        front_cc = (cc[1] > 0.25) & (cc[1] < 0.75) & (cc[2] < 0.25)
        middle_cc = (
            (cc[1] >= 0.25) & (cc[1] <= 0.75) & (cc[2] >= 0.25) & (cc[2] <= 0.75)
        )
        back_cc = (cc[1] > 0.25) & (cc[1] < 0.75) & (cc[2] > 0.75)
        top_front_cc = (cc[1] > 0.75) & (cc[2] < 0.25)
        top_middle_cc = (cc[1] > 0.75) & (cc[2] > 0.25) & (cc[2] < 0.75)
        top_back_cc = (cc[1] > 0.75) & (cc[2] > 0.75)
        self.cell_idx: List[np.ndarray] = [
            bottom_front_cc,
            bottom_middle_cc,
            bottom_back_cc,
            front_cc,
            middle_cc,
            back_cc,
            top_front_cc,
            top_middle_cc,
            top_back_cc,
        ]

        # Get list of boundary indices
        fc = self.g3d.face_centers
        bottom_front_bc = (fc[1] < 0.25) & (fc[2] < 0.25)
        bottom_middle_bc = (fc[1] < 0.25) & (fc[2] > 0.25) & (fc[2] < 0.75)
        bottom_back_bc = (fc[1] < 0.25) & (fc[2] > 0.75)
        front_bc = (fc[1] > 0.25) & (fc[1] < 0.75) & (fc[2] < 0.25)
        middle_bc = (
            (fc[1] >= 0.25) & (fc[1] <= 0.75) & (fc[2] >= 0.25) & (fc[2] <= 0.75)
        )
        back_bc = (fc[1] > 0.25) & (fc[1] < 0.75) & (fc[2] > 0.75)
        top_front_bc = (fc[1] > 0.75) & (fc[2] < 0.25)
        top_middle_bc = (fc[1] > 0.75) & (fc[2] > 0.25) & (fc[2] < 0.75)
        top_back_bc = (fc[1] > 0.75) & (fc[2] > 0.75)
        self.bc_idx: List[np.ndarray] = [
            bottom_front_bc,
            bottom_middle_bc,
            bottom_back_bc,
            front_bc,
            middle_bc,
            back_bc,
            top_front_bc,
            top_middle_bc,
            top_back_bc,
        ]

        # Save exact forms as attributes of the class
        x, y, z = sym.symbols("x y z")
        self.alpha = x - 0.5
        self.beta1 = y - 0.25
        self.beta2 = y - 0.75
        self.gamma1 = z - 0.25
        self.gamma2 = z - 0.75
        self.n = 1.5
        self.dist_bottom_front = (
            self.alpha ** 2 + self.beta1 ** 2 + self.gamma1 ** 2
        ) ** 0.5
        self.dist_bottom_middle = (self.alpha ** 2 + self.beta1 ** 2) ** 0.5
        self.dist_bottom_back = (
            self.alpha ** 2 + self.beta1 ** 2 + self.gamma2 ** 2
        ) ** 0.5
        self.dist_front = (self.alpha ** 2 + self.gamma1 ** 2) ** 0.5
        self.dist_middle = (self.alpha ** 2) ** 0.5
        self.dist_back = (self.alpha ** 2 + self.gamma2 ** 2) ** 0.5
        self.dist_top_front = (
            self.alpha ** 2 + self.beta2 ** 2 + self.gamma1 ** 2
        ) ** 0.5
        self.dist_top_middle = (self.alpha ** 2 + self.beta2 ** 2) ** 0.5
        self.dist_top_back = (
            self.alpha ** 2 + self.beta2 ** 2 + self.gamma2 ** 2
        ) ** 0.5
        self.bubble = (
            self.beta1 ** 2 * self.beta2 ** 2 * self.gamma1 ** 2 * self.gamma2 ** 2
        )

    def __repr__(self) -> str:
        return "Exact solution  object for 3D problem"

    def p3d(self, which="cc"):
        """
        Computation of the exact pressure in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to which.

        """

        cc = self.g3d.cell_centers
        x, y, z = sym.symbols("x y z")

        p3_bottom_front = self.dist_bottom_front ** (1 + self.n)
        p3_bottom_middle = self.dist_bottom_middle ** (1 + self.n)
        p3_bottom_back = self.dist_bottom_back ** (1 + self.n)
        p3_front = self.dist_front ** (1 + self.n)
        p3_middle = self.dist_middle ** (1 + self.n) + self.bubble * self.dist_middle
        p3_back = self.dist_back ** (1 + self.n)
        p3_top_front = self.dist_top_front ** (1 + self.n)
        p3_top_middle = self.dist_top_middle ** (1 + self.n)
        p3_top_back = self.dist_top_back ** (1 + self.n)
        p3_sym: list = [
            p3_bottom_front,
            p3_bottom_middle,
            p3_bottom_back,
            p3_front,
            p3_middle,
            p3_back,
            p3_top_front,
            p3_top_middle,
            p3_top_back,
        ]

        p3_fun = [sym.lambdify((x, y, z), p, "numpy") for p in p3_sym]
        p3_cc = np.zeros(self.g3d.num_cells)
        for (p, idx) in zip(p3_fun, self.cell_idx):
            p3_cc += p(cc[0], cc[1], cc[2]) * idx

        if which == "sym":
            return p3_sym
        elif which == "fun":
            return p3_fun
        else:
            return p3_cc

    def gradp3d(self, which="cc"):
        """
        Computation of the exact pressure gradients in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to which.

        """
        x, y, z = sym.symbols("x y z")
        cc = self.g3d.cell_centers

        gradp3_sym = [
            [sym.diff(p, x), sym.diff(p, y), sym.diff(p, z)]
            for p in self.p3d(which="sym")
        ]

        gradp3_fun = [
            [
                sym.lambdify((x, y, z), gradp[0], "numpy"),
                sym.lambdify((x, y, z), gradp[1], "numpy"),
                sym.lambdify((x, y, z), gradp[2], "numpy"),
            ]
            for gradp in gradp3_sym
        ]

        gradp3x_cc = np.zeros(self.g3d.num_cells)
        gradp3y_cc = np.zeros(self.g3d.num_cells)
        gradp3z_cc = np.zeros(self.g3d.num_cells)
        for (u, idx) in zip(gradp3_fun, self.cell_idx):
            gradp3x_cc += u[0](cc[0], cc[1], cc[2]) * idx
            gradp3y_cc += u[1](cc[0], cc[1], cc[2]) * idx
            gradp3z_cc += u[2](cc[0], cc[1], cc[2]) * idx
        gradp3_cc = np.array([gradp3x_cc, gradp3y_cc, gradp3z_cc])

        if which == "sym":
            return gradp3_sym
        elif which == "fun":
            return gradp3_fun
        else:
            return gradp3_cc

    def u3d(self, which="cc"):
        """
        Computation of the exact fluxes in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to which.

        """

        x, y, z = sym.symbols("x y z")
        cc = self.g3d.cell_centers

        u3_sym = [
            [-sym.diff(p, x), -sym.diff(p, y), -sym.diff(p, z)]
            for p in self.p3d(which="sym")
        ]

        u3_fun = [
            [
                sym.lambdify((x, y, z), u[0], "numpy"),
                sym.lambdify((x, y, z), u[1], "numpy"),
                sym.lambdify((x, y, z), u[2], "numpy"),
            ]
            for u in u3_sym
        ]

        ux_cc = np.zeros(self.g3d.num_cells)
        uy_cc = np.zeros(self.g3d.num_cells)
        uz_cc = np.zeros(self.g3d.num_cells)
        for (u, idx) in zip(u3_fun, self.cell_idx):
            ux_cc += u[0](cc[0], cc[1], cc[2]) * idx
            uy_cc += u[1](cc[0], cc[1], cc[2]) * idx
            uz_cc += u[2](cc[0], cc[1], cc[2]) * idx
        u3_cc = np.array([ux_cc, uy_cc, uz_cc])

        if which == "sym":
            return u3_sym
        elif which == "fun":
            return u3_fun
        else:
            return u3_cc

    def f3d(self, which="cc"):
        """
        Computation of the exact sources in the bulk

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to which.

        """
        x, y, z = sym.symbols("x y z")
        cc = self.g3d.cell_centers

        f3_sym = [
            sym.diff(u[0], x) + sym.diff(u[1], y) + sym.diff(u[2], z)
            for u in self.u3d(which="sym")
        ]

        f3_fun = [sym.lambdify((x, y, z), f, "numpy") for f in f3_sym]

        f3_cc = np.zeros(self.g3d.num_cells)
        for (f, idx) in zip(f3_fun, self.cell_idx):
            f3_cc += f(cc[0], cc[1], cc[2]) * idx

        if which == "sym":
            return f3_sym
        elif which == "fun":
            return f3_fun
        else:
            return f3_cc

    def lmbda(self, which="cc"):
        """
        Computation of the exact mortar flux

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (y, z).

        Returns
        -------
        Appropiate object according to which.

        """
        y, z = sym.symbols("y z")
        cc = self.mg.cell_centers

        lmbda_sym = self.bubble

        lmbda_fun = sym.lambdify((y, z), lmbda_sym, "numpy")

        lmbda_cc = lmbda_fun(cc[1], cc[2])

        if which == "sym":
            return lmbda_sym
        elif which == "fun":
            return lmbda_fun
        else:
            return lmbda_cc

    def p2d(self, which="cc"):
        """
        Computation of the exact fracture pressure

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (y, z).

        Returns
        -------
        Appropiate object according to which.

        """
        y, z = sym.symbols("y z")
        cc = self.g2d.cell_centers

        # Symbolic expression
        p2_sym = -1 * self.bubble

        # Function
        p2_fun = sym.lambdify((y, z), p2_sym, "numpy")

        # Cell center pressures (in real coordinates)
        p2_cc = p2_fun(cc[1], cc[2])

        if which == "sym":
            return p2_sym
        elif which == "fun":
            return p2_fun
        else:
            return p2_cc

    def gradp2d(self, which="cc"):
        """
        Computation of the exact fracture pressure gradient

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (y, z).

        Returns
        -------
        Appropiate object according to which.

        """
        y, z = sym.symbols("y z")
        cc = self.g2d.cell_centers

        gradp2_sym = [
            sym.diff(self.p2d("sym"), y),
            sym.diff(self.p2d("sym"), z),
        ]

        gradp2_fun = [
            sym.lambdify((y, z), gradp2_sym[0], "numpy"),
            sym.lambdify((y, z), gradp2_sym[1], "numpy"),
        ]

        gradp2_cc = np.array([gradp2_fun[0](cc[1], cc[2]), gradp2_fun[1](cc[1], cc[2])])

        if which == "sym":
            return gradp2_sym
        elif which == "fun":
            return gradp2_fun
        else:
            return gradp2_cc

    def u2d(self, which="cc"):
        """
        Computation of the exact fracture velocity

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (y, z).

        Returns
        -------
        Appropiate object according to which.

        """
        y, z = sym.symbols("y z")
        cc = self.g2d.cell_centers

        u2_sym = [
            -sym.diff(self.p2d("sym"), y),
            -sym.diff(self.p2d("sym"), z),
        ]

        u2_fun = [
            sym.lambdify((y, z), u2_sym[0], "numpy"),
            sym.lambdify((y, z), u2_sym[1], "numpy"),
        ]

        u2_cc = np.array([u2_fun[0](cc[1], cc[2]), u2_fun[1](cc[1], cc[2])])

        if which == "sym":
            return u2_sym
        elif which == "fun":
            return u2_fun
        else:
            return u2_cc

    def f2d(self, which="cc"):
        """
        Computation of the exact fracture source

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (y).

        Returns
        -------
        Appropiate object according to which.

        """
        y, z = sym.symbols("y z")
        cc = self.g2d.cell_centers

        jump_lmbda_sym = 2 * self.lmbda("sym")
        div_u2_sym = sym.diff(self.u2d("sym")[0], y) + sym.diff(self.u2d("sym")[1], z)
        f2_sym = div_u2_sym - jump_lmbda_sym

        f2_fun = sym.lambdify((y, z), f2_sym, "numpy")

        f2_cc = f2_fun(cc[1], cc[2])

        if which == "sym":
            return f2_sym
        elif which == "fun":
            return f2_fun
        else:
            return f2_cc

    def dir_bc_values(self):
        """
        Computation of the exact Dirichlet data for the bulk

        Returns
        -------
        NumPy array containing the pressure boundary values

        """

        # Get face-center coordinates
        fc = self.g3d.face_centers

        # Initialize boundary values array
        bc_values = np.zeros(self.g3d.num_faces)

        # Evaluate exact pressure at external boundary faces at each region
        for (p, idx) in zip(self.p3d("fun"), self.bc_idx):
            bc_values[idx] = p(fc[0][idx], fc[1][idx], fc[2][idx])

        return bc_values

    def integrate_f3d(self):

        # Declare integration method and get hold of elements in QuadPy format
        int_method = qp.t3.get_good_scheme(5)
        elements = utils.get_quadpy_elements(self.g3d, self.g3d)

        integral = np.zeros(self.g3d.num_cells)
        for (f, idx) in zip(self.f3d("fun"), self.cell_idx):
            # Declare integrand
            def integrand(x):
                return f(x[0], x[1], x[2]) * np.ones_like(x[0])

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    def integrate_f2d(self):

        method = qp.t2.get_good_scheme(10)
        g_rot = mdestimates.RotatedGrid(self.g2d)
        elements = utils.get_quadpy_elements(self.g2d, g_rot)
        elements *= -1  # we have to use the real y coordinates here
        # TODO: CHECK IF THIS TRICK STILL WORKS FOR 3D

        def integrand(x):
            return self.f2d("fun")(x[0], x[1])

        integral = method.integrate(integrand, elements)

        return integral
