import numpy as np
import operator

nLambda = 3
nInter = 4

'''Declaration and set up variables for RFO'''


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def construct_T(drdq):
    T = np.matrix(np.zeros((nInter, nInter)))
    drdq = drdq.getT()

    for i in range(nLambda):
        for j in range(nInter):
            T[i, j] = drdq[i, j]
    for i in range(nInter - nLambda):
        T[i + nLambda, i] = 1;

    T = gram_schmidt_columns(T)

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
        norm += float(i) * float(i);
    norm = norm ** (0.5)

    return norm


def construct_h(dEdq, drdq):
    lambdas = construct_lambda(drdq, dEdq)
    h = np.matrix(np.zeros((1, nInter)))

    h = dEdq - drdq * lambdas
    return h


'''****************************************************************************************************************
*****************************************************Start RFO*****************************************************
****************************************************************************************************************'''


def bfgs_update_W(prev_W, prev_grad_W, delta_q, dEdq, drdq):
    s_k = delta_q
    y_k = construct_h(dEdq, drdq)

    tmp1 = y_k * y_k.getT()
    tmp2 = y_k.getT() * s_k
    tmp3 = prev_W * s_k * s_k.getT() * prev_W
    tmp4 = s_k.getT() * prev_W * s_k

    W = prev_W - tmp3 / tmp4 + tmp1 / tmp2

    return W

def consrtruct_AH(hess, grad):
    Z = np.matrix(np.zeros((1, nInter - nLambda)))
    Hess = np.bmat([[Z.getT(), grad.getT()], [grad, hess]])

    return Hess


def get_eigen_vect(AH):
    eig_val, eig_vec = np.linalg.eig(AH)
    return eig_val, eig_vec


def sort_eigen(eig_val, eig_vec):
    dict_eig = {eig_val[i]: eig_vec[i] for i in range(nInter - nLambda + 1)}
    dict_eig = sorted(dict_eig.items(), key=operator.itemgetter(0))
    return dict_eig


def calculate_delat_q(sorted_eig):
    return 1

def interation(delta_y, red_grad, delta_x, beta):
    k_max = 100

    x_beta = 1.0
    y_beta = 1.0
    g_beta = 1.0

    step_retsriction = beta * max(y_beta * min(x_beta, g_beta), 1 / 10)

    dy = beta
    g = beta
    dx = beta

    sf = 2 ** (0.5)

    norm_deltay = compute_norm(delta_y)
    norm_deltax = compute_norm(delta_x)
    norm_red_grad = compute_norm(red_grad)

    for i in range(k_max):
        if dy < 0.75 * norm_deltay or norm_deltay < 10 ** (-2):
            y_beta = min(2, y_beta * sf)
        elif dy > 1.25 * norm_deltay and dy >= 10 ** (-5):
            y_beta = max(1 / 10, y_beta / sf)

        if i != k_max:
            if dx < norm_deltax < beta:
                x_beta = min(1, x_beta * sf)
            elif dx > 1.25 * norm_deltax or dx >= beta:
                x_beta = max(1 / 5, x_beta * sf)
                dx = norm_deltax

        if g < 0.75 * norm_red_grad and g < beta:
            g_beta = min(1, g_beta * sf)
        elif g > 1.25 * norm_red_grad or g >= beta:
            g_beta = max(1 / 5, g_beta * sf)
            g = norm_red_grad

    return 1


def main():
    #   Установим точку в которой мы стоим (compute energy)
    q = np.array([1, 2, 3, 4])
    q = q.reshape(4, 1)
    print("q")
    print(q)

    dq = np.matrix(np.zeros((1, nInter)))
    dq = dq.getT()

    for i in range(nInter):
        dq[i] = 19 + i * 3.4

    #    dq = np.array([1, 2, 3, 4])
    #    dq = q.reshape(4, 1)
    print("dq")
    print(dq)

    #   Вектор-столбец ограничений
    r = np.array([1, 2, 3])
    r = r.reshape(3, 1)
    print("r")
    print(r)


    #   Производная от ограничений
    drdq = np.matrix('5 3 7; 3 1 4; 7 9 7; 0 2 6')

    drdq = gram_schmidt_columns(drdq)
    print("drdq")
    print(drdq)

    #   Сделано поттому что нужен вектор столбец
    dEdq = np.array([1, 2, 3, 4])
    dEdq = dEdq.reshape(4, 1)
    print("dEdq")
    print(dEdq)

    W = np.matrix('1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16')
    print("W")
    print(W)

    T = construct_T(drdq)
    print("T")
    print(T)

    T_b = construct_T_b(T)
    print("T_b")
    print(T_b)

    T_ti = construct_T_ti(T)
    print("T_ti")
    print(T_ti)

    lambdas = construct_lambda(drdq, dEdq)
    print("lambdas")
    print(lambdas)

    dy = construct_dy(drdq, T_b, r)
    print("dy")
    print(dy)

    x = construct_x(q, T_ti)
    print("x")
    print(x)

    dx = construct_dx(dq, T_ti)
    print("dx")
    print(dx)

    red_grad = construct_reduced_grad(dEdq, W, dy, T_b, T_ti)
    print("reduced grad")
    print(red_grad)

    red_hess = construct_reduced_hess(W, T_ti)
    print("reduced hess")
    print(red_hess)

    h = construct_h(dEdq, drdq)
    print("h")
    print(h)

    AH = consrtruct_AH(red_hess, red_grad)
    print("AH")
    print(AH)

    eig_val, eig_vec = get_eigen_vect(AH)
    print("eig_val")
    print(eig_val)
    print("eig_vec")
    print(eig_vec)

    print("sorted_eig")
    print(sort_eigen(eig_val, eig_vec))

    print("updated_W")
    print(bfgs_update_W(W, h, dq, dEdq, drdq))

if __name__ == '__main__':
    main()
