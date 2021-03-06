import numpy as np


def create_q(q):
    q_vec = []
    for i in q:
        for j in i.pos:
            q_vec.append(j)
    return np.array(q_vec)


def create_charge(q):
    q_charge = []
    for i in q:
        q_charge.append(i.charge)
    return q_charge


def write_config(charge, vect):
    n = len(charge)

    string = ""
    for i in range(n):
        string = string + str(charge[i]) + "  "
        string = string + str(vect[3 * i]) + "   " + str(vect[3 * i + 1]) + "   " + str(vect[3 * i + 2]) + "\n"

    print(string)
