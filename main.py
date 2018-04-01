import read_constrains as cnstr_r
import numpy as np


def create_q(q):
    q_vec = []
    for i in q:
        for j in i.pos:
            q_vec.append(j)
    return np.array(q_vec)


start_config = cnstr_r.parser_xyz("/home/rusanov-vs/PycharmProjects/constr/in/struct.xyz")
constr = cnstr_r.constrains("/home/rusanov-vs/PycharmProjects/constr/in/cnstr", start_config)

print(constr.get_constr_grad(create_q(start_config)))
# print(cnstr_r.A_func(constr.CONSTR_LIST[0], create_q_start(start_config)).get_grad())
# print(cnstr_r.A_func(constr.CONSTR_LIST[0], create_q_start(start_config)).get_hess())

a = np.array([[1, 0], [0, 1]])
b = np.array([[1, 1]])
print(np.dot(2, 1))
