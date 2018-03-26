"""
This module contain constrained RFO optimisation
ARTICLES:
1)  J. Chem. Theory Comput. 2005, 1, 1029-1037
    New General Tools for Constrained Geometry Optimizations (Luca De Vico, Massimo Olivucci, Roland Lindh)
2)  J Comput. Chem. 18: 992-1003, 1997
    A Reduced-Restricted-Quasi-Newton-Raphson Method for Locating and Optimizing
    Energy Crossing Points Between Two Potential Energy Surfaces (Josep Maria Anglada, Josep Maria Bofill)
3)  Theor. Chem. Acc. (1998) 100:265-274
    On the automatic restricted-step rational-function-optimization method (Emili Besalu, Josep Maria Bofill)
"""

from math import *
import os
import numpy as np
from matplotlib import pyplot as plt
import time as t

# Count of constrain
nLambda = 1
# Count of degrees of freedom
nInter = 2


class Func:
    def __init__(self):
        pass

    def __call__(self, x):
        x, y = x
        return (1 - y ** 2) * x ** 2 * exp(-x ** 2) + .5 * y ** 2

    def grad(self, x):
        x, y = x[0, 0], x[1, 0]

        dEdx = 2 * (1 - y ** 2) * exp(-x ** 2) * (x - x ** 3)
        dEdy = (1 - 2 * x ** 2 * exp(-x ** 2)) * y
        return np.matrix([[dEdx], [dEdy]])

    def hess(self, x):
        x, y = x[0, 0], x[1, 0]

        xx = 2 * (1 - y ** 2) * exp(-x ** 2) * (2 * x ** 4 - 5 * x ** 2 + 1)
        xy = 4 * y * exp(-x ** 2) * x * (x - 1) * (x + 1);
        yy = 1 - 2 * x ** 2 * exp(-x ** 2)

        return np.matrix([[xx, xy], [xy, yy]])


class Func3:
    def __init__(self):
        self.inner = Func()

    def __call__(self, x):
        return self.inner(x[:2]) + self.inner(x[1:])

    def grad(self, x):
        grad = np.matrix(np.zeros((3, 1)))
        grad[:2, 0] = self.inner.grad(x[:2])
        grad[1:, 0] = self.inner.grad(x[1:])

        return grad

    def hess(self, x):
        hess = np.matrix(np.zeros((3, 3)))
        hess[:2, :2] = self.inner.hess(x[:2])
        hess[1:, 1:] = self.inner.hess(x[1:])

        return hess


# def r(x):
#     x, y = x
#     return sqrt(x ** 2 + y ** 2) - 0.5
#
#
# def get_drdq(x):
#     x, y = x[0, 0], x[1, 0]
#     x = x / sqrt(x ** 2 + y ** 2)
#     y = y / sqrt(x ** 2 + y ** 2)
#
#     return np.matrix([[x], [y]])
#
#
# def get_d2rdq2(x):
#     x, y = x[0, 0], x[1, 0]
#     xx = y ** 2 / sqrt(x ** 2 + y ** 2) ** 3
#     xy = - x * y / sqrt(x ** 2 + y ** 2) ** 3
#     yy = x ** 2 / sqrt(x ** 2 + y ** 2) ** 3
#
#     return np.matrix([[xx, xy], [xy, yy]])

def r(x):
    x, y = x
    return (1 - y ** 2) * x ** 2 * exp(-x ** 2) + .5 * y ** 2 + 1

def get_drdq(x):
    x, y = x[0, 0], x[1, 0]

    dEdx = (2 * (1 - y ** 2) * exp(-x ** 2) * (x - x ** 3))
    dEdy = (1 - 2 * x ** 2 * exp(-x ** 2)) * y
    return np.matrix([[dEdx], [dEdy]])

def get_d2rdq2(x):
    x, y = x[0, 0], x[1, 0]

    xx = (2 * (1 - y ** 2) * exp(-x ** 2) * (2 * x ** 4 - 5 * x ** 2 + 1))
    xy = (4 * y * exp(-x ** 2) * x * (x - 1) * (x + 1))
    yy = -(1 - 2 * x ** 2 * exp(-x ** 2))

    return np.matrix([[xx, xy], [xy, yy]])


'''Declaration and set up variables for RFO'''


def gram_schmidt_columns(x):
    """
    Gram-Schmidt orthogonalization
        returns matrix with orthonormal columns
    :param x: matrix of vectors to make Gram-Schmidt  procedure
    :return q: orthogonalized system of vectors
    """
    q, r_tmp = np.linalg.qr(x)
    return q


def get_rfo_step(grad, hess, beta):
    """
    Compute RFO step
    :param grad: gradient of Q-function (contains information about constrains)
    :param hess: hessian of Q-function (contains information about constrains)
    :param beta: step restriction and multiplier for S-matrix
    :return: RFO step
    """
    m = np.matrix(np.block([[hess, grad], [grad.getT(), 0]]))
    print("jhsadj\n", m, "\n")
    z1 = np.matrix(np.zeros((1, nInter - nLambda)))
    z2 = np.matrix(np.ones((1, 1)))
    s = np.matrix(np.eye(nInter - nLambda, nInter - nLambda))
    s = np.bmat([[z2, z1.getT()], [z1, beta * s]])
    m = s.getI() * m

    print(m)
    w, v = np.linalg.eigh(m)
    print(w)
    dx = v[:-1, 0] / v[-1, 0]
    return dx


def construct_t_matrix(drdq):
    """
    Make T_matrix to divide space into constrained an non-constrained part
        returns T_matrix
    :param drdq: gradient of constrains
    :return T: matrix with orthogonalized system of vectors
    """
    T = np.matrix(np.zeros((nInter, nInter)))
    drdq = drdq.getT()

    # Fill rows of T_matrix with gradient of constrains
    T[:nLambda] = drdq

    # Fill zero-part of T with ones on the diagonal
    for index in range(nInter - nLambda):
        T[index + nLambda, index] = 1

    T = gram_schmidt_columns(T.getT()).getT()

    return T


def construct_T_b(T):
    """
    Divide matrix T with two sub-spaces
        returns T_b matrix contained constrained part
    :param T: T -- np.matrix with orthogonalized system of vect
    :return T_b: Matrix contain part of the space with constrains
    """
    T_b = np.matrix(np.zeros((nLambda, nInter)))

    for i in range(nLambda):
        for j in range(nInter):
            T_b[i, j] = T[i, j]

    T_b = T_b.getT()
    return T_b


def construct_T_ti(T):
    """
    Divide matrix T with two sub-spaces
        returns T_b matrix contained non-constrained part
    :param T: T -- np.matrix with orthogonalized system
    :return T_ti: Matrix contain part of the space connected with
    constrains and value of energy
    """
    T_ti = np.matrix(np.zeros((nInter - nLambda, nInter)))
    for i in range(nInter - nLambda):
        for j in range(nInter):
            T_ti[i, j] = T[i + nLambda, j]

    T_ti = T_ti.getT()

    return T_ti


def construct_dy(drdq, T_b, r):
    """
    Makes a projection into T_b space
        returns vector-projection
    :param drdq: vector of gradients of constrains at the special point
    :param T_b: matrix contained non-constrained part
    :param r: value of constrains at the special point
    :return dy: step in the constrained space
    """
    tmp = drdq.getT() * T_b
    tmp = tmp.getI()
    dy = tmp * r
    dy *= -1

    return dy


def construct_x(q, T_ti):
    """
    Makes a projection into T_ti space
        returns vector-projection
    :param q: vector of special point
    :param T_ti: Matrix contain part of the space connected with
    constrains and value of energy
    :return x: step in the space connected with constrained part of space and energy
    """
    x = T_ti.getT() * q
    return x


def construct_dx(dq, T_ti):
    """
    Makes a projection of derivatives into T_ti space
        returns vector-projection
    :param dq: delta vector of special point
    :param T_ti:  Matrix contain part of the space connected with
    constrains and value of energy
    :return x: delta step in the space connected with constrained part of space and energy
    """
    dx = T_ti.getT() * dq
    return dx


def construct_reduced_grad(dEdq, W, dy, T_b, T_ti):
    """
    Build gradient Q-fucntion, optimisation of Q-function is a solution of Lagrange
        problem
        returns reduced gradient of Q function (contains information about constrains)
    :param dEdq: gradient of energy at a special point
    :param W: hessian of the Lagrange problem
    :param dy: step in the constrained space
    :param T_b: Matrix contain part of the space with constrains
    :param T_ti: Matrix contain part of the space connected with
    constrains and value of energy
    :return red_grad: reduced gradient of Q function (contains information about constrains)
    """

    tmp1 = dEdq
    tmp2 = W * T_b
    tmp3 = tmp2 * dy
    tmp1 = -tmp3 + tmp1
    red_grad = T_ti.getT() * tmp1

    return red_grad


def construct_reduced_hess(W, T_ti):
    """
    Build hessian Q-function, optimisation of Q-function is a solution of Lagrange
        problem
        returns reduced hessian of Q function
    :param W: hessian of the Lagrange problem
    :param T_ti: Matrix contain part of the space connected with
    constrains and value of energy
    :return red_hess: reduced hessian of Q function (contains information about constrains)
    """
    tmp = T_ti.getT() * W
    red_hess = tmp * T_ti

    return red_hess


def construct_lambda(drdq, dEdq):
    """
    Build vector of lambdas
        returns vector of lambdas
    :param drdq: gradient of constrains at a special point
    :param dEdq: gradient of energy at a special point
    :return rLambda: vector of Lagrange multipliers
    """
    tmp = drdq.getT() * drdq
    tmp.getI()
    tmp = tmp * drdq.getT()
    rLambda = tmp * dEdq

    return rLambda


def compute_norm(vect_col):
    """
    Calculate normal of vector-column
        returns float normal
    :param vect_col: vector
    :return: norm of the vector
    """
    return np.linalg.norm(vect_col, ord=2)


def construct_h(dEdq, drdq, lambdas):
    """
    Build h-vector (gradient of W)
        return gradient vector W
    :param dEdq: gradient of energy at a special point
    :param drdq: gradient of constrains at a special point
    :param lambdas: vector of Lagrange multipliers
    :return h: gradient of W (part of redused gradient and hessian of Q-function,
    optimisation of Q-function is a solution of Lagrange problem
    """
    h = dEdq - drdq * lambdas
    return h


def bfgs_update_W(prev_W, delta_grad, delta_q):
    """
    BFGS update of W, W - is the part of reduced gradient and hessian
        we mean B_s in BFGS formula is W
        returns updated W
    :param prev_W: value W in the previous step
    :param delta_grad: change of gradient W (h-vector) between previous step
    :param delta_q: change of reaction coordinate between previous step
    :return W: updated with BFGS procedure W
    """
    s_k = delta_q
    y_k = delta_grad

    tmp1 = y_k * y_k.getT()
    tmp2 = y_k.getT() * s_k
    tmp3 = prev_W * s_k * s_k.getT() * prev_W
    tmp4 = s_k.getT() * prev_W * s_k

    W = prev_W - tmp3 / tmp4 + tmp1 / tmp2

    return W


def plot(xs, ys, func):
    """
    Debug function, build plot of RFO steps on the model f(x, y) energy
    :param xs: list of abscissas
    :param ys: list of ordinates
    :param func: model 2D potential
    :return: plot

    ..warning:: should to close plt.close() and clear field plt.clf() after every call

    ..note:: for showing plot after the function need to usr plt.show()
    """
    plt.figure(figsize=(10, 10))

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    x_grid, y_grid = np.meshgrid(x, y)

    z_E = np.zeros_like(x_grid)
    for i, (row_x, row_y) in enumerate(zip(x_grid, y_grid)):
        for j, (x, y) in enumerate(zip(row_x, row_y)):
            z_E[i, j] = func([x, y])

    z_r = np.zeros_like(x_grid)
    for i, (row_x, row_y) in enumerate(zip(x_grid, y_grid)):
        for j, (x, y) in enumerate(zip(row_x, row_y)):
            z_r[i, j] = r([x, y])

    # plt.figure(figsize=(50, 50))
    plt.contour(x_grid, y_grid, z_E - z_r)
    plt.contour(x_grid, y_grid, z_r, [0], colors='black')
    plt.plot(xs, ys, c='r')


# DEBUG INFO: collect statistics of steps of RFO before convergence, execution time, count of converges
list_steps = []
list_times = []
list_conv = []


def rfo_constr(q_start, use_beta=True, beta=1.5, trust_radius=1):
    """
    This function find constrained local minimum of energy
    :param q_start: start geometry (start point on the potential energy surface)
    :param use_beta: parameter to switch on / off usage of beta like step restriction multiplier
    and multiplier for S-matrix in RFO
    :param beta: step restriction multiplier and also and multiplier for S-matrix in RFO
    :param trust_radius: if use_beta = false is using like step restriction
    :type q_start: np.matrix()
    :type trust_radius: float
    :type use_beta: bool
    :type beta: float

    :returns: result of function is a point on the potential energy surface which is local minimum with
    respects to constrains

    ..warning:: function is not stable with respect to the choice beta we do not know exactly how to
    stabilise behavior with any beta

    .. note:: know function includes some debug tools like drawing plots and store execution
    time and count of converged local minimum information
    """
    # DEBUG INFO: check start time
    start_time = t.time()

    # Add test energy function
    E = Func()

    # History of all previous points
    step_history = []
    # History of all previous gradients of energy
    dEdq_history = []
    # History of all previous gradients of constrains
    drdq_history = []

    # Add start point to the history
    step_history.append(q_start)

    # DEBUG INFO: coordinates of the path to build a plot
    xs = []
    ys = []

    plot(xs, ys, E)
    plt.show()

    # Initialization of internal parameters
    x_beta = 1.0
    y_beta = 1.0
    g_beta = 1.0

    # Maximum count of the steps
    k_max = 1000

    # Get energy hessian at start point
    d2Edq2 = E.hess(q_start)
    d2rdq2 = get_d2rdq2(q_start)

    # Main loop of the optimisation with constraint
    for i in range(k_max):
        # DEBUG INFO: collect trip for plot it
        xs.append(step_history[i][0, 0])
        ys.append(step_history[i][1, 0])

        # plot(xs, ys, E)
        # if not os.path.exists("/home/rusanov-vs/PycharmProjects/constr/pic/pic_path" + str(len(list_steps) + 1)):
        #     os.makedirs("/home/rusanov-vs/PycharmProjects/constr/pic/pic_path" + str(len(list_steps) + 1))
        #
        # plt.savefig("/home/rusanov-vs/PycharmProjects/constr/pic/pic_path" + str(len(list_steps) + 1) + "/" + "step"
        #  + str(i) + ".png",
        #             format='png', dpi=100)
        # plt.clf()
        # plt.close()

        # Compute gradients of energy and constrains
        dEdq = E.grad(step_history[i])
        drdq = get_drdq(step_history[i])

        # Add dEdq and drdq to the history
        dEdq_history.append(dEdq)
        drdq_history.append(drdq)

        # Compute current lambda
        lam = construct_lambda(drdq, dEdq)

        # Compute W at start_point with new current lambdas
        W = d2Edq2 + lam[0, 0] * d2rdq2

        # Make series of BFGS update from start_point to the current point
        for j in range(len(step_history) - 1):
            W = bfgs_update_W(W, construct_h(dEdq_history[j + 1], drdq_history[j + 1], lam) -
                              construct_h(dEdq_history[j], drdq_history[j], lam),
                              step_history[j + 1] - step_history[j])

        # Dividing into to subspaces
        T = construct_t_matrix(drdq)
        T_b = construct_T_b(T)
        T_ti = construct_T_ti(T)

        # Calculate step in T_b subspace
        delta_y = construct_dy(drdq, T_b, r(step_history[i]))

        # Compute norm of step in constrained space
        norm_deltay = compute_norm(delta_y)

        # Reduced gradient and reduced hessian of the main optimisation Q-function
        grad = construct_reduced_grad(dEdq, W, delta_y, T_b, T_ti)
        norm_red_grad = compute_norm(grad)
        hess = construct_reduced_hess(W, T_ti)

        # Beta feedback mechanism switch
        if use_beta:
            step_rest = beta * max(y_beta * min(x_beta, g_beta), 1 / 10)
            delta_x = get_rfo_step(grad, hess, 1 / step_rest)

            dy = beta
            g = beta
            dx = beta

            sf = 2 ** (0.5)

            norm_deltax = compute_norm(delta_x)

            # Make mysterious actions with set up step restriction
            if dy < 0.75 * norm_deltay or norm_deltay < 10 ** (-2):
                y_beta = min(2, y_beta * sf)
            elif dy > 1.25 * norm_deltay and dy >= 10 ** (-5):
                y_beta = max(1 / 10, y_beta / sf)
                dy = norm_deltay

            if i != k_max:
                if dx < norm_deltax < beta:
                    x_beta = min(1, x_beta * sf)
                elif dx > 1.25 * norm_deltax or dx >= beta:
                    x_beta = max(1 / 5, x_beta / sf)
                    dx = norm_deltax

            if g < 0.75 * norm_red_grad and g < beta:
                g_beta = min(1, g_beta * sf)
            elif g > 1.25 * norm_red_grad or g >= beta:
                g_beta = max(1 / 5, g_beta / sf)
                g = norm_red_grad

            # Cut the step in case of big step because of bad derivatives
            if norm_deltax > step_rest:
                delta_x = delta_x / norm_deltax
                delta_x = delta_x * step_rest

            # Cut the step in case of big step because of bad derivatives
            if norm_deltay > step_rest:
                delta_y = delta_y / norm_deltay
                delta_y = delta_y * step_rest

        else:
            delta_x = get_rfo_step(grad, hess, 1)

            # Cut the step in case of big step because of bad derivatives
            norm_deltax = compute_norm(delta_x)
            if norm_deltax > trust_radius:
                delta_x = delta_x / norm_deltax
                delta_x = delta_x * trust_radius

            # Cut the step in case of big step because of bad derivatives
            if norm_deltay > trust_radius:
                delta_y = delta_y / norm_deltay
                delta_y = delta_y * trust_radius

        # Calculate step from rfo optimisation
        step = T_b * delta_y + T_ti * delta_x
        # Construct coordinates of new point int the energy coordinates
        new_point = step_history[i] + step
        # Add new point to history of points
        step_history.append(new_point)

        # primitive stop-criteria
        if norm_red_grad < 10 ** (-6):
            # DEBUG INFO: collect count of steps and information about convergence
            list_steps.append(i)
            list_conv.append(1)
            break

    # DEBUG INFO: check up execution time
    t_end = t.time() - start_time
    list_times.append(t_end)

    # # DEBUG INFO: show trip in console
    # for stp in step_history:
    #     print('x = {}, y = {}'.format(stp[0, 0], stp[1, 0]))

    # DEBUG INFO: show plot of the trip on the potential
    plot(xs, ys, E)
    plt.savefig(
        "/home/rusanov-vs/PycharmProjects/constr/pic/" + str((len(list_steps))) + ".png",
        format='png', dpi=100)
    plt.clf()
    plt.close()


# DEBUG INFO: start point debug
q_start = np.matrix([[1.], [1.]])

for i in range(1):
    rfo_constr(q_start, beta=1)
    print("Done,", i)

sum = 0
for i in range(len(list_steps)):
    sum = sum + list_steps[i]
print('AVERAGE STEP = {}, MEDIAN STEP  = {}'.format(sum / len(list_steps), list_steps[int(len(list_steps) / 2)]))

sum = 0
for i in range(len(list_times)):
    sum = sum + list_times[i]
print('AVERAGE TIME = {}, MEDIAN TIME = {}'.format(sum / len(list_times), list_times[int(len(list_times) / 2)]))

sum = 0
for i in range(len(list_conv)):
    sum = sum + list_conv[i]
print('COUNT OF CONV = {}'.format(sum))

