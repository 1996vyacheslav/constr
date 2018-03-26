import os
import numpy as np


def rad_to_grad(angle):
    return angle * np.pi / 180.0


def calc_B(atom_1, atom_2):
    return np.linalg.norm(atom_1 - atom_2, ord=2)


def calc_A(atom_1, atom_2, atom_3):
    tmp1 = atom_1 - atom_2
    tmp2 = atom_3 - atom_2
    angle = np.arccos(tmp1.dot(tmp2) / (np.linalg.norm(tmp1, ord=2) * np.linalg.norm(tmp2, ord=2)))
    return rad_to_grad(angle)


def calc_D(atom_1, atom_2, atom_3, atom_4):
    tmp1 = atom_1 - atom_2
    tmp2 = atom_3 - atom_2
    tmp3 = atom_4 - atom_3

    angle = np.arctan2(((tmp1 * tmp2) * (tmp2 * tmp3)).dot(tmp2 / np.linalg.norm(tmp2, ord=2)),
                       (tmp1 * tmp2).dot(tmp2 * tmp3))

    return rad_to_grad(angle)


class constrain:
    constr_type = ""
    constr_param = ""
    n_atoms = []
    length = .0
    angle = .0

    def __init__(self, type="", param="", atoms=[], len=.0):
        self.constr_type = type
        self.constr_param = param
        self.n_atoms = atoms
        self.length = len

        if self.constr_param == "F":
            if self.constr_type == "B":
                self.length = calc_B(self.n_atoms[0], self.n_atoms[1])
            elif self.constr_type == "A":
                self.angle = calc_A(self.n_atoms[0], self.n_atoms[1], self.n_atoms[2])
            elif self.constr_type == "D":
                self.angle = calc_D(self.n_atoms[0], self.n_atoms[1], self.n_atoms[3], self.n_atoms[4])


class constrains:
    COUNT_OF_CNSTR = 0

    def __init__(self, cnstr_path_file, start_config):
        self.file_path = cnstr_path_file
        self.start_config = start_config

    def read_file(self):
        if not os.path.exists(self.file_path):
            print("NOT FOUND: File with constrains")
        else:
            list_lines = []

            line_B = []
            line_A = []
            line_D = []
            file = open(self.file_path, 'r')

            for line in file:
                list_lines.append(line)

            self.COUNT_OF_CNSTR = int(list_lines[0])

            for line in list_lines:
                if line[0] == 'B':
                    line_B.append(line)
                elif line[0] == 'A':
                    line_A.append(line)
                elif line[0] == 'D':
                    line_D.append(line)

            file.close()
        return 1
