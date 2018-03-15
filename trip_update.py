from math import *
import numpy as np
from matplotlib import pyplot as plt
import time as t

nLambda = 1
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


def r(x):
    return np.linalg.norm(x) - .3


def get_drdq(x):
    return 2 * x


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def get_rfo_step(grad, hess, beta):
    M = np.matrix(np.block([[hess, grad], [grad.getT(), 0]]))

    Z1 = np.matrix(np.zeros((1, nInter - nLambda)))
    Z2 = np.matrix(np.ones((1, 1)))
    S = np.matrix(np.eye(nInter - nLambda, nInter - nLambda))
    S = np.bmat([[Z2, Z1.getT()], [Z1, beta * S]])
    M = S.getI() * M

    w, v = np.linalg.eigh(M)
    dx = v[:-1, 0] / v[-1, 0]
    return dx


def construct_T(drdq):
    T = np.matrix(np.zeros((nInter, nInter)))
    drdq = drdq.getT()

    T[:nLambda] = drdq

    for i in range(nInter - nLambda):
        T[i + nLambda, i] = 1;

    T = gram_schmidt_columns(T.getT()).getT()

    return T


def construct_T_b(T):
    T_b = np.matrix(np.zeros((nLambda, nInter)))
    for i in range(nLambda):
        for j in range(nInter):
            T_b[i, j] = T[i, j]
    T_b = T_b.getT()
    return T_b


def construct_T_ti(T):
    T_ti = np.matrix(np.zeros((nInter - nLambda, nInter)))
    for i in range(nInter - nLambda):
        for j in range(nInter):
            T_ti[i, j] = T[i + nLambda, j]

    T_ti = T_ti.getT()

    return T_ti


def construct_dy(drdq, T_b, r):
    tmp = drdq.getT() * T_b
    tmp = tmp.getI()
    dy = tmp * r
    dy *= -1

    return dy


def construct_x(q, T_ti):
    x = T_ti.getT() * q
    return x


def construct_dx(dq, T_ti):
    dx = T_ti.getT() * dq
    return dx


def construct_reduced_grad(dEdq, W, dy, T_b, T_ti):
    tmp1 = dEdq
    tmp2 = W * T_b
    tmp3 = tmp2 * dy
    tmp1 = -tmp3 + tmp1
    red_grad = T_ti.getT() * tmp1

    return red_grad


def construct_reduced_hess(W, T_ti):
    tmp = T_ti.getT() * W
    red_hess = tmp * T_ti

    return red_hess


def construct_lambda(drdq, dEdq):
    tmp = drdq.getT() * drdq
    tmp.getI()
    tmp = tmp * drdq.getT()
    rLambda = tmp * dEdq

    return rLambda


def compute_norm(vect_col):
    norm = 0
    for i in vect_col:
        norm += float(i) * float(i)
    norm = norm ** (0.5)

    return norm


def construct_h(dEdq, drdq, lambdas):
    h = dEdq - drdq * lambdas
    return h


def bfgs_update_W(prev_W, delta_grad, delta_q):
    s_k = delta_q
    y_k = delta_grad

    tmp1 = y_k * y_k.getT()
    tmp2 = y_k.getT() * s_k
    tmp3 = prev_W * s_k * s_k.getT() * prev_W
    tmp4 = s_k.getT() * prev_W * s_k

    W = prev_W + tmp3 / tmp4 - tmp1 / tmp2

    return W


def plot(xs, ys, func):
    plt.figure(figsize=(10, 10))

    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1, 2, 100)
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
    plt.contour(x_grid, y_grid, z_E, 200)
    plt.contour(x_grid, y_grid, z_r, [0], colors='black')
    plt.plot(xs, ys, c='r')


def alg_trip(trust_radius, use_beta, beta=0):
    list_W_min_W_upt = []

    # Check start time
    start_time = t.time()

    # Add test energy function
    E = Func()

    # History of all previous points
    step_history = []
    # History of all previous gradients of energy
    dEdq_history = []
    # History of all previous gradients of constrains
    drdq_history = []

    # Start point
    q_start = np.matrix([[1.], [1.]])

    # Add start point to the history
    step_history.append(q_start)

    # Coordinates of the path to build a plot
    xs = []
    ys = []

    # Initialization of internal parameters
    x_beta = 1.0
    y_beta = 1.0
    g_beta = 1.0

    # Maximum count of the steps
    k_max = 100

    # Get energy hessian at start point
    d2Edq2 = E.hess(q_start)

    # Main loop of the optimisation with constraint
    for i in range(k_max):
        xs.append(step_history[i][0, 0])
        ys.append(step_history[i][1, 0])

        # Compute gradients of energy and constrains
        dEdq = E.grad(step_history[i])
        drdq = get_drdq(step_history[i])

        # Add dEdq and drdq to the history
        dEdq_history.append(dEdq)
        drdq_history.append(drdq)

        # Compute current lambda
        lam = construct_lambda(drdq, dEdq)

        # Compute W at start_point with new current lambdas
        W = d2Edq2 + lam[0, 0] * 2 * np.identity(2)

        # Make series of BFGS update from start_point to the current point
        for j in range(len(step_history) - 1):
            W = bfgs_update_W(W, construct_h(dEdq_history[j + 1], drdq_history[j + 1], lam) -
                              construct_h(dEdq_history[j], drdq_history[j], lam),
                              step_history[j + 1] - step_history[j])

        # DEBUG
        W_rel = E.hess(step_history[i]) + lam[0, 0] * 2 * np.identity(2)
        print(W_rel - W)

        # Dividing into to subspaces
        T = construct_T(drdq)
        T_b = construct_T_b(T)
        T_ti = construct_T_ti(T)

        # Calculate step in T_b subspace
        delta_y = construct_dy(drdq, T_b, r(step_history[i]))

        # Cut the step in case of big step because of bad derivatives
        norm_deltay = compute_norm(delta_y)
        if norm_deltay > trust_radius:
            delta_y = delta_y / norm_deltay
            delta_y = delta_y * trust_radius

        # Reduced gradient and reduced hessian of the main optimisation Q-function
        grad = construct_reduced_grad(dEdq, W, delta_y, T_b, T_ti)
        hess = construct_reduced_hess(W, T_ti)

        # Beta feedback mechanism switch
        if use_beta:
            step_rest = beta * max(y_beta * min(x_beta, g_beta), 1 / 10)
            delta_x = get_rfo_step(grad, hess, step_rest)

            dy = beta
            g = beta
            dx = beta

            sf = 2 ** (0.5)

            norm_deltax = compute_norm(delta_x)
            norm_red_grad = compute_norm(grad)

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
        else:
            delta_x = get_rfo_step(grad, hess, 1)

        # Cut the step in case of big step because of bad derivatives
        norm_deltax = compute_norm(delta_x)
        if norm_deltax > trust_radius:
            delta_x = delta_x / norm_deltax
            delta_x = delta_x * trust_radius

        # Calculate step from rfo optimisation
        step = T_b * delta_y + T_ti * delta_x
        # Construct coordinates of new point int the energy coordinates
        new_point = step_history[i] + step
        # Add new point to history of points
        step_history.append(new_point)

    # Check execution time
    t_end = t.time() - start_time
    print("Time", t_end)

    for stp in step_history:
        print('x = {}, y = {}'.format(stp[0, 0], stp[1, 0]))

    plot(xs, ys, E)
    plt.show()


alg_trip(0.3, True, 4)
