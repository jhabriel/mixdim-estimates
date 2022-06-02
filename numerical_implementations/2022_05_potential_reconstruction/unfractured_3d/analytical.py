import mdestimates
import porepy as pp
import numpy as np
import sympy as sym
import quadpy as qp
import mdestimates.estimates_utils as utils


class ExactSolution:
    def __init__(self, gb: pp.GridBucket):
        """
        Class containing the exact solution for the flow problem

        Parameters
        ----------
        gb (GridBucket): Containing computed topological information
        """

        # Grid bucket
        self.gb: pp.GridBucket = gb

        # Grid and its dictionary
        self.g: pp.Grid = self.gb.grids_of_dimension(3)[0]
        self.d: dict = self.gb.node_props(self.g)

    def __repr__(self) -> str:
        return "Exact solution  object for 3D unfractured problem"

    def p(self, which="cc"):
        """
        Computation of the exact pressure solution

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to "which".

        """

        cc = self.g.cell_centers
        x, y, z = sym.symbols("x y z")

        # p_sym = sym.sin(2 * sym.pi * x) * sym.sin(2 * sym.pi * y) * sym.sin(2 * sym.pi * z)
        p_sym = x * (1 - x) * y * (1 - y) * z * (1 - z)
        p_fun = sym.lambdify((x, y, z), p_sym, "numpy")
        p_cc = p_fun(cc[0], cc[1], cc[2])

        if which == "sym":
            return p_sym
        elif which == "fun":
            return p_fun
        else:
            return p_cc

    def gradp(self, which="cc"):
        """
        Computation of the pressure gradient

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to "which".

        """
        x, y, z = sym.symbols("x y z")
        cc = self.g.cell_centers

        gradp_sym = [
            sym.diff(self.p("sym"), x),
            sym.diff(self.p("sym"), y),
            sym.diff(self.p("sym"), z),
        ]

        gradp_fun = [
            sym.lambdify((x, y, z), gradp_sym[0], "numpy"),
            sym.lambdify((x, y, z), gradp_sym[1], "numpy"),
            sym.lambdify((x, y, z), gradp_sym[2], "numpy"),
        ]

        gradpx_cc = gradp_fun[0](cc[0], cc[1], cc[2])
        gradpy_cc = gradp_fun[1](cc[0], cc[1], cc[2])
        gradpz_cc = gradp_fun[2](cc[0], cc[1], cc[2])
        gradp_cc = np.array([gradpx_cc, gradpy_cc, gradpz_cc])

        if which == "sym":
            return gradp_sym
        elif which == "fun":
            return gradp_fun
        else:
            return gradp_cc

    def u(self, which="cc"):
        """
        Computation of the exact Darcy velocity

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to "which".

        """

        x, y, z = sym.symbols("x y z")
        k = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # permeability = identity matrix
        cc = self.g.cell_centers

        u_sym_x = (
                - k[0][0] * self.gradp("sym")[0]
                - k[0][1] * self.gradp("sym")[1]
                - k[0][2] * self.gradp("sym")[2]
        )

        u_sym_y = (
                - k[1][0] * self.gradp("sym")[0]
                - k[1][1] * self.gradp("sym")[1]
                - k[1][2] * self.gradp("sym")[2]
        )

        u_sym_z = (
                - k[2][0] * self.gradp("sym")[0]
                - k[2][1] * self.gradp("sym")[1]
                - k[2][2] * self.gradp("sym")[2]
        )

        u_sym = [u_sym_x, u_sym_y, u_sym_z]

        u_fun = [
            sym.lambdify((x, y, z), u_sym[0], "numpy"),
            sym.lambdify((x, y, z), u_sym[1], "numpy"),
            sym.lambdify((x, y, z), u_sym[2], "numpy"),
        ]

        ux_cc = u_fun[0](cc[0], cc[1], cc[2])
        uy_cc = u_fun[1](cc[0], cc[1], cc[2])
        uz_cc = u_fun[2](cc[0], cc[1], cc[2])
        u_cc = np.array([ux_cc, uy_cc, uz_cc])

        if which == "sym":
            return u_sym
        elif which == "fun":
            return u_fun
        else:
            return u_cc

    def f(self, which="cc"):
        """
        Computation of the exact source term

        Parameters
        ----------
        which (str): "cc" for the cell center evaluation, "sym" for the
            symbolic expressions, and "fun" for the lambda functions
            depending on (x, y, z).

        Returns
        -------
        Appropiate object according to "which".

        """
        x, y, z = sym.symbols("x y z")
        cc = self.g.cell_centers

        f_sym = (
                sym.diff(self.u("sym")[0], x)
                + sym.diff(self.u("sym")[1], y)
                + sym.diff(self.u("sym")[2], z)
        )

        f_fun = sym.lambdify((x, y, z), f_sym, "numpy")

        f_cc = f_fun(cc[0], cc[1], cc[2])

        if which == "sym":
            return f_sym
        elif which == "fun":
            return f_fun
        else:
            return f_cc

    def dir_bc_values(self) -> np.ndarray:
        """
        Computation of the Dirichlet boundary condtions from exact pressure

        Returns
        -------
        bc_values: NumPy array containing the pressure boundary values

        """

        # Get face-center coordinates
        fc = self.g.face_centers

        # Evaluate exact pressure at external boundary faces
        bc_values = np.zeros(self.g.num_faces)
        idx = self.g.get_boundary_faces()
        bc_values[idx] = self.p("fun")(fc[0][idx], fc[1][idx], fc[2][idx])

        return bc_values

    def integrate_f(self, degree: int = 10) -> np.ndarray:
        """
        Computes the numerical integration of the source term

        Parameters
        ----------
        degree (int)

        Returns
        -------
        integrated_source: Numpy array containing the integrated sources

        """

        # Declare integration method and get hold of elements in quadpy format
        int_method = qp.t3.get_good_scheme(degree)
        elements = utils.get_quadpy_elements(self.g, self.g)

        def integrand(x):
            return self.f("fun")(x[0], x[1], x[2]) * np.ones_like(x[0])

        # Integrate
        integral = int_method.integrate(integrand, elements)

        return integral
