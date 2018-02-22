import numpy as np

nLambda = 3
nInter = 4


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


def main():
    #   Установим точку в которой мы стоим (compute energy)
    q = np.array([1, 2, 3, 4])
    q = q.reshape(4, 1)
    print("q")
    print(q)

    dq = np.array([1, 2, 3, 4])
    dq = q.reshape(4, 1)
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

    l = construct_lambda(drdq, dEdq)
    print("lambda")
    print(l)

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

if __name__ == '__main__':
    main()
