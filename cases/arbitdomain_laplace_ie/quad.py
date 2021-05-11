import numpy as np
import numpy.linalg as la
import scipy.special
import matplotlib.pyplot as plt

# import scipy.sparse.linalg as spla

cos = np.cos
sin = np.sin
pi = np.pi


def curve(t):
    a = 1.0
    n = 5
    eps = 0.25
    return np.array(
        [
            (a + eps * a * cos(n * t)) * cos(t),
            (a + eps * a * cos(n * t)) * sin(t),
        ]
    )


def dcurve_dt(t):
    a = 1.0
    n = 5
    eps = 0.25
    return np.array(
        [
            -a * sin(t) - eps * a * (n * sin(n * t) * cos(t) + cos(n * t) * sin(t)),
            a * cos(t) - eps * a * (n * sin(n * t) * sin(t) - cos(n * t) * sin(t)),
        ]
    )


def u_exact(points):
    x, y = points
    # return (-1.0 / (2 * np.pi)) * np.log(np.sqrt((x - 1.0) ** 2 + (y - 1.0) ** 2))
    return y / np.sqrt(x ** 2 + y ** 2)


# grid_size = 25
# xd = np.linspace(-1.5, 1.5, grid_size)
# XD, YD = np.meshgrid(xd, xd)
# test_targets = np.zeros((2, grid_size ** 2))
# test_targets[0] = XD.reshape(-1)
# test_targets[1] = YD.reshape(-1)
# test_targets = np.array([[-0.2, 0], [0.2, 0], [0, -0.2], [0, 0.2]]).T

grid_size_r = 25
grid_size_t = 100
theta = np.linspace(0, 2.0 * np.pi, grid_size_t)
a = 1.0
n = 5
eps = 0.25
r_bc = a + eps * a * cos(n * theta)
r = np.linspace(0.01, 1, grid_size_r)
R = np.outer(r, r_bc)
T = np.outer(np.ones_like(r), theta)
# convert to x and y
XD = R * cos(T)
YD = R * sin(T)
test_targets = np.zeros((2, grid_size_t * grid_size_r))
test_targets[0] = XD.reshape(-1)
test_targets[1] = YD.reshape(-1)

npanels = 40
# This data structure helps you get started by setting up geometry
# and Gauss quadrature panels for you.


class QuadratureInfo:
    def __init__(self, nintervals):
        self.nintervals = nintervals
        # par_length = 2*np.pi
        intervals = np.linspace(0, 2 * np.pi, nintervals + 1)
        self.npoints = 7 + 1
        self.shape = (nintervals, self.npoints)

        ref_info = scipy.special.legendre(self.npoints).weights
        ref_nodes = ref_info[:, 0]
        ref_weights = ref_info[:, 2]

        par_intv_length = intervals[1] - intervals[0]

        self.par_nodes = np.zeros((nintervals, self.npoints))
        for i in range(nintervals):
            a, b = intervals[i : i + 2]

            assert abs((b - a) - par_intv_length) < 1e-10
            self.par_nodes[i] = ref_nodes * par_intv_length * 0.5 + (b + a) * 0.5

        self.curve_nodes = curve(self.par_nodes.reshape(-1)).reshape(2, nintervals, -1)
        self.curve_deriv = dcurve_dt(self.par_nodes.reshape(-1)).reshape(
            2, nintervals, -1
        )

        self.curve_speed = la.norm(self.curve_deriv, 2, axis=0)

        tangent = self.curve_deriv / self.curve_speed
        tx, ty = tangent
        self.normals = np.array([ty, -tx])

        self.curve_weights = self.curve_speed * ref_weights * par_intv_length / 2
        self.panel_lengths = np.sum(self.curve_weights, 1)

        # if 0:
        plt.plot(self.curve_nodes[0].reshape(-1), self.curve_nodes[1].reshape(-1), "x-")
        plt.quiver(
            self.curve_nodes[0], self.curve_nodes[1], self.normals[0], self.normals[1]
        )
        # plt.show()
