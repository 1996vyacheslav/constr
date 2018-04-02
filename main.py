import read_constrains as cnstr_r
import utils
import Logger
import sys
import numpy as np
import rfo_constr_arr as rfo_c
import gaussian_wrapper


class energy_gauss:
    def __init__(self, gaussian, charges):
        self.gauss = gaussian
        self.charge = charges

    def __call__(self, x):
        return self.gauss(self.charge, x)

    def grad(self, x):
        two_in_one = self.gauss.value_grad(self.charge, x)
        tmp = []
        for i in two_in_one[1]:
            tmp.append([i])
        return np.array(tmp)

    def hess(self, x):
        two_in_one = self.gauss.value_grad_hess(self.charge, x)
        return two_in_one[2]


sys.stdout = Logger.Logger("/home/rusanov-vs/PycharmProjects/constr/rfo_constr_arr_log.txt")

start_config = cnstr_r.parser_xyz("/home/rusanov-vs/PycharmProjects/constr/in/struct.xyz")

constr = cnstr_r.constrains("/home/rusanov-vs/PycharmProjects/constr/in/cnstr", start_config)

# LOGGING INFORMATION
print("----------CONSTRAINS----------")
for i in constr.CONSTR_LIST:
    print(i)


start = utils.create_q(start_config)
charge = utils.create_charge(start_config)

func = energy_gauss(gaussian_wrapper.GaussianWrapper(n_proc=3, mem=1000), charge)

# LOGGING INFORMATION
print("----------START OPTIMISATION----------")
utils.write_config(charge, start)

if rfo_c.rfo_constr(Energy_Func=func, nInter=len(start), nLambda=len(constr.CONSTR_LIST),
                    q_start=start, charge=charge, constrains=constr, use_beta=True, beta=1.5):
    print("----------END OPTIMISATION----------")

# class Func:
#     def __init__(self):
#         pass
#
#     def __call__(self, x):
#         x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
#         return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
#
#     def grad(self, x):
#         x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
#         return np.array([[1 + x1], [1 + x2], [1 + x3], [1 + x4],
#                          [1 + x5], [1 + x6], [1 + x7], [1 + x8], [1 + x9]])
#
#     def hess(self, x):
#         m = np.zeros((len(x), len(x)))
#         m[:, :] = [0]
#         return m
