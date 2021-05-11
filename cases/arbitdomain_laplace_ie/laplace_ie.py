import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
# from quad import QuadratureInfo, u_exact
# from quad import npanels, test_targets
from sympy import symbols, log, lambdify, diff, sqrt
from mpmath import pi, factorial
from scipy.sparse.linalg import gmres
# from quad import XD, YD


plt.gca().set_aspect("equal")


# gmres iteration counter
# https://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1


counter = gmres_counter()


# evaluates QBX coeffs of Laplace kernel with given order and ext/int
# lambdified in order tau, x(1, 2), y(1, 2), nx(1, 2), ny(1, 2), r
def eval_QBX(order):
    tau, x_1, y_1, x_2, y_2 = symbols('tau, x_1, y_1, x_2, y_2')
    nx_1, nx_2, ny_1, ny_2, r = symbols('nx_1, nx_2, ny_1, ny_2, r')
    tau_1 = x_1 + r * (1 - tau) * nx_1
    tau_2 = x_2 + r * (1 - tau) * nx_2
    # Greens function for 2D
    kernel = -log(sqrt((tau_1 - y_1) ** 2 + (tau_2 - y_2) ** 2)) / 2 / pi
    # print(kernel)
    single_layer = lambdify([tau, x_1, x_2, y_1, y_2,
                            nx_1, nx_2, ny_1, ny_2, r], kernel)
    green_normal_der = ny_1 * diff(kernel, y_1) + ny_2 * diff(kernel, y_2)
    # print(green_normal_der)
    double_layer = lambdify([tau, x_1, x_2, y_1, y_2,
                            nx_1, nx_2, ny_1, ny_2, r], green_normal_der)
    # calculating derivs of double layer only here
    qbx_exp = green_normal_der
    for i in range(1, order + 1):
        deriv = diff(green_normal_der, tau, i) / factorial(i)
        qbx_exp += deriv
    exp_term = lambdify([tau, x_1, x_2, y_1, y_2,
                        nx_1, nx_2, ny_1, ny_2, r], qbx_exp)
    return single_layer, double_layer, exp_term


def bvp(n, domain, exact_test):
    # domain = QuadratureInfo(n)
    normals_x, normals_y = domain.normals.reshape(2, -1)
    nodes_x, nodes_y = domain.curve_nodes.reshape(2, -1)
    # taking exp_radius as panel_length / 2 from QBX paper
    qbx_radius = np.repeat(domain.panel_lengths, domain.npoints) / 2
    total_points = nodes_x.shape[0]
    normal_mat_x = np.broadcast_to(normals_x, (total_points, total_points))
    normal_mat_y = np.broadcast_to(normals_y, (total_points, total_points))
    node_mat_x = np.broadcast_to(nodes_x, (total_points, total_points))
    node_mat_y = np.broadcast_to(nodes_y, (total_points, total_points))
    radius_mat = np.broadcast_to(qbx_radius, (total_points, total_points)).T
    # take care of normal signs here
    D_qbx_int = qbx_exp(0, node_mat_x.T, node_mat_y.T, node_mat_x, node_mat_y,
                        -normal_mat_x.T, -normal_mat_y.T, normal_mat_x,
                        normal_mat_y,
                        radius_mat) * domain.curve_weights.reshape(-1)
    D_qbx_ext = qbx_exp(0, node_mat_x.T, node_mat_y.T, node_mat_x, node_mat_y,
                        normal_mat_x.T, normal_mat_y.T, normal_mat_x,
                        normal_mat_y,
                        radius_mat) * domain.curve_weights.reshape(-1)
    test_density = np.ones((total_points, 1))
    print(np.average((D_qbx_int + D_qbx_ext) * 0.5 @ test_density))
    rhs = exact_test.reshape(-1)
    # averaging interior exterior limits
    A = (D_qbx_int + D_qbx_ext) * 0.5 - 0.5 * np.identity(total_points)
    soln_density, msg = gmres(A, rhs, tol=1e-13, callback=counter)
    print("GMRES iter:", counter.niter)
    return soln_density.reshape(n, -1)


def eval_DLP(targets, sources, weights, source_normals, density):
    normals_x, normals_y = source_normals.reshape(2, -1)
    nodes_x, nodes_y = sources.reshape(2, -1)
    target_number = targets.shape[1]
    total_points = nodes_x.shape[0]
    test_normal_mat_x = np.broadcast_to(normals_x,
                                        (target_number, total_points))
    test_normal_mat_y = np.broadcast_to(normals_y,
                                        (target_number, total_points))
    sources_mat_x = np.broadcast_to(nodes_x,
                                    (target_number, total_points))
    sources_mat_y = np.broadcast_to(nodes_y,
                                    (target_number, total_points))
    targets_mat_x = np.broadcast_to(targets[0],
                                    (total_points, target_number)).T
    targets_mat_y = np.broadcast_to(targets[1],
                                    (total_points, target_number)).T
    D = (dp(1, targets_mat_x, targets_mat_y, sources_mat_x, sources_mat_y,
         0, 0, test_normal_mat_x, test_normal_mat_y, 0) * weights.reshape(-1))
    DLP_eval = D @ density.reshape(-1)
    return DLP_eval


sp, dp, qbx_exp = eval_QBX(4)
# domain = QuadratureInfo(npanels)
# soln1 = bvp(npanels)
# # soln2 = bvp(npanels2)
# # testing against targets
# exact_test = u_exact(test_targets)
# num_test = eval_DLP(test_targets, domain.curve_nodes, domain.curve_weights,
#                     domain.normals, soln1)
# err = num_test - exact_test
# plt.contourf(XD, YD, num_test.reshape(XD.shape[0], -1), levels=np.linspace(-1, 1, 100), cmap="inferno")
# plt.colorbar()
# plt.show()
# print(la.norm(err))
