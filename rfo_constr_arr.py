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

import numpy as np
import utils
import time as t
from numpy.linalg import inv


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


def get_rfo_step(grad, hess, beta, nInter, nLambda):
    """
    Compute RFO step
    :param grad: gradient of Q-function (contains information about constrains)
    :param hess: hessian of Q-function (contains information about constrains)
    :param beta: step restriction and multiplier for S-matrix
    :param nInter: count of variables
    :param nLambda: count of constrains
    :return: RFO step
    """
    m = np.block([[hess, grad], [grad.transpose(), 0]])
    z1 = np.zeros((1, nInter - nLambda))
    z2 = np.ones((1, 1))
    s = np.eye(nInter - nLambda, nInter - nLambda)
    s = np.block([[z2, z1], [z1.transpose(), beta * s]])
    m = np.dot(inv(s), m)

    w, v = np.linalg.eigh(m)

    dx = v[:-1, 0] / v[-1, 0]

    list_tmp = []
    for i in dx:
        list_tmp.append([i])
    dx = np.array(list_tmp)
    return dx


def construct_t_matrix(drdq, nInter, nLambda):
    """
    Make T_matrix to divide space into constrained an non-constrained part
        returns T_matrix
    :param drdq: gradient of constrains
    :param nInter: count of variables
    :param nLambda: count of constrains
    :return T: matrix with orthogonalized system of vectors
    """
    T = np.zeros((nInter, nInter))
    drdq = drdq.transpose()
    # Fill rows of T_matrix with gradient of constrains
    T[:nLambda] = drdq

    # Fill zero-part of T with ones on the diagonal
    for index in range(nInter - nLambda):
        T[index + nLambda, index] = 1

    T = gram_schmidt_columns(T.transpose()).transpose()

    return T


def construct_T_b(T, nInter, nLambda):
    """
    Divide matrix T with two sub-spaces
        returns T_b matrix contained constrained part
    :param T: T -- np.matrix with orthogonalized system of vect
    :param nInter: count of variables
    :param nLambda: count of constrains
    :return T_b: Matrix contain part of the space with constrains
    """
    T_b = np.zeros((nLambda, nInter))

    for i in range(nLambda):
        for j in range(nInter):
            T_b[i, j] = T[i, j]

    T_b = T_b.transpose()
    return T_b


def construct_T_ti(T, nInter, nLambda):
    """
    Divide matrix T with two sub-spaces
        returns T_b matrix contained non-constrained part
    :param T: T -- np.matrix with orthogonalized system
    :param nInter: count of variables
    :param nLambda: count of constrains
    :return T_ti: Matrix contain part of the space connected with
    constrains and value of energy
    """
    T_ti = np.zeros((nInter - nLambda, nInter))
    for i in range(nInter - nLambda):
        for j in range(nInter):
            T_ti[i, j] = T[i + nLambda, j]

    T_ti = T_ti.transpose()

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
    tmp = np.dot(drdq.transpose(), T_b)
    tmp = inv(tmp)
    dy = np.dot(tmp, r)
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
    x = np.dot(T_ti.transpose(), q)
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
    dx = np.dot(T_ti.transpose(), dq)
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
    tmp2 = np.dot(W, T_b)
    tmp3 = np.dot(tmp2, dy)
    tmp1 = -tmp3 + tmp1
    red_grad = np.dot(T_ti.transpose(), tmp1)

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
    tmp = np.dot(T_ti.transpose(), W)
    red_hess = np.dot(tmp, T_ti)

    return red_hess


def construct_lambda(drdq, dEdq):
    """
    Build vector of lambdas
        returns vector of lambdas
    :param drdq: gradient of constrains at a special point
    :param dEdq: gradient of energy at a special point
    :return rLambda: vector of Lagrange multipliers
    """
    tmp = np.dot(drdq.transpose(), drdq)
    tmp = inv(tmp)
    tmp = np.dot(tmp, drdq.transpose())
    rLambda = np.dot(tmp, dEdq)

    return rLambda


def compute_norm(vect_col):
    """
    Calculate normal of vector-column
        returns float normal
    :param vect_col: vector
    :return: norm of the vector
    """
    return np.linalg.norm(vect_col)


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
    h = dEdq - np.dot(drdq, lambdas)
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

    tmp1 = np.dot(y_k, y_k.transpose())
    tmp2 = np.dot(y_k.transpose(), s_k)
    tmp3 = np.dot(np.dot(np.dot(prev_W, s_k), s_k.transpose()), prev_W)
    tmp4 = np.dot(np.dot(s_k.transpose(), prev_W), s_k)

    W = prev_W - tmp3 / tmp4 + tmp1 / tmp2

    return W


def rfo_constr(nInter, nLambda, q_start, constrains, Energy_Func, charge, use_beta=True, beta=1.5, trust_radius=1):
    """
    This function find constrained local minimum of energy
    :param nInter: count of variables
    :param nLambda: count of constrains
    :param q_start: start geometry (start point on the potential energy surface)
    :param constrains: object of class constrains with all constrains
    :param Energy_Func: energy from Gaussian or other program
    :param charge: contains list of charges
    :param use_beta: parameter to switch on / off usage of beta like step restriction multiplier
    and multiplier for S-matrix in RFO
    :param beta: step restriction multiplier and also and multiplier for S-matrix in RFO
    :param trust_radius: if use_beta = false is using like step restriction

    :type nInter: int
    :type nLambda: int
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
    E = Energy_Func

    # History of all previous points
    step_history = []
    # History of all previous gradients of energy
    dEdq_history = []
    # History of all previous gradients of constrains
    drdq_history = []

    # Initialization of internal parameters
    x_beta = 1.0
    y_beta = 1.0
    g_beta = 1.0

    # Maximum count of the steps
    k_max = 10000

    # Get energy hessian at start point
    d2Edq2 = E.hess(q_start)
    d2rdq2 = constrains.get_constr_val(q_start)

    # Add start point to the history
    step_history.append(q_start)

    # Main loop of the optimisation with constraint
    for i in range(k_max):
        # Compute gradients of energy and constrains
        dEdq = E.grad(step_history[i])
        drdq = constrains.get_constr_grad(step_history[i])


        # Add dEdq and drdq to the history
        dEdq_history.append(dEdq)
        drdq_history.append(drdq)

        # Compute current lambda
        lam = construct_lambda(drdq, dEdq)

        # Compute W at start_point with new current lambdas
        summa = 0
        for j in range(len(d2rdq2)):
            summa = summa + lam[0, 0] * d2rdq2[j]

        W = d2Edq2 + summa

        # Make series of BFGS update from start_point to the current point
        for j in range(len(step_history) - 1):
            W = bfgs_update_W(W, construct_h(dEdq_history[j + 1], drdq_history[j + 1], lam) -
                              construct_h(dEdq_history[j], drdq_history[j], lam),
                              step_history[j + 1] - step_history[j])

        # Dividing into to subspaces
        T = construct_t_matrix(drdq, nInter, nLambda)
        T_b = construct_T_b(T, nInter, nLambda)
        T_ti = construct_T_ti(T, nInter, nLambda)

        # Calculate step in T_b subspace
        delta_y = construct_dy(drdq, T_b, constrains.get_constr_val(step_history[i]))

        # Compute norm of step in constrained space
        norm_deltay = compute_norm(delta_y)

        # Reduced gradient and reduced hessian of the main optimisation Q-function
        grad = construct_reduced_grad(dEdq, W, delta_y, T_b, T_ti)
        norm_red_grad = compute_norm(grad)
        hess = construct_reduced_hess(W, T_ti)

        # Beta feedback mechanism switch
        if use_beta:
            step_rest = beta * max(y_beta * min(x_beta, g_beta), 1 / 10)
            delta_x = get_rfo_step(grad, hess, 1 / step_rest, nInter, nLambda)

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
        step = np.dot(T_b, delta_y) + np.dot(T_ti, delta_x)

        tmp = []
        for k in step:
            tmp.append(float(k))
        step = np.array(tmp)

        # Construct coordinates of new point int the energy coordinates
        new_point = step_history[i] + step

        # LOGING INFORMATION
        print("STEP NUMBER = {}\nGRADIENT_NORM = {}".format(i + 1, norm_red_grad))
        utils.write_config(charge, new_point)

        # Add new point to history of points
        step_history.append(new_point)

        # primitive stop-criteria
        if norm_red_grad < 10 ** (-8):
            # LOGING INFORMATION
            print("END OF OPTIMISATION")
            utils.write_config(charge, step_history[i])
            break

    # DEBUG INFO: check up execution time
    t_end = t.time() - start_time
    return 1
