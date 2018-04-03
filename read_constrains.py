import os
import numpy as np


def string_ready_list(string):
    not_tuple = ('', '\n')
    string = string.strip('\n')
    string_list = string.split(' ')
    ready_list = []

    for i in string_list:
        if i not in not_tuple:
            ready_list.append(i)

    return ready_list


def parser_xyz(file_path):
    list_atoms = []
    if not os.path.exists(file_path):
        print("NOT FOUND: File with structure (*.xyz)")
    else:
        file = open(file_path, 'r')

        num_in_file = 0
        for line in file:
            line = string_ready_list(line)
            charge = 0
            comp_pos = []

            for i in range(len(line)):
                if i == 0:
                    charge = int(line[i])
                else:
                    comp_pos.append(float(line[i]))
            num_in_file = num_in_file + 1
            list_atoms.append(atom(charge, comp_pos, num_in_file))
        return list_atoms


def rad_to_grad(angle):
    return angle / np.pi * 180.0


def calc_B(atom_1, atom_2):
    return np.linalg.norm(atom_1 - atom_2)


def calc_A(atom_1, atom_2, atom_3):
    eps = 10 ** (-8)
    tmp1 = atom_1 - atom_2
    tmp2 = atom_3 - atom_2
    tmp4 = tmp1.dot(tmp2) / (np.linalg.norm(tmp1) * np.linalg.norm(tmp2))

    if tmp4 + eps > -1 and tmp4 - eps < -1:
        return 180.
    elif tmp4 + eps > 1 and tmp4 - eps < 1:
        return 0.
    else:
        angle = np.arccos(tmp4)
        return rad_to_grad(angle)


def calc_D(atom_1, atom_2, atom_3, atom_4):
    tmp1 = atom_1 - atom_2
    tmp2 = atom_3 - atom_2
    tmp3 = atom_3 - atom_4

    angle = np.arctan2(np.cross(np.cross(tmp1, tmp2), np.cross(tmp2, tmp3)).dot(tmp2 / np.linalg.norm(tmp2)),
                       np.cross(tmp1, tmp2).dot(np.cross(tmp2, tmp3)))

    angle = rad_to_grad(angle)
    return angle


def pacer_B(string, mol):
    at = []
    list_splited_str = string_ready_list(string)

    if list_splited_str[-1] == "F":
        at.append(mol[int(list_splited_str[1]) - 1])
        at.append(mol[int(list_splited_str[2]) - 1])
        constr = constrain(type="B", param="F", atoms=at)
    else:
        at.append(mol[int(list_splited_str[1]) - 1])
        at.append(mol[int(list_splited_str[2]) - 1])
        constr = constrain(type="B", atoms=at, len=float(list_splited_str[-1]))

    return constr


def parcer_A(string, mol):
    at = []
    list_splited_str = string_ready_list(string)

    if list_splited_str[-1] == "F":
        for i in range(3):
            at.append(mol[int(list_splited_str[i + 1]) - 1])
        constr = constrain(type="A", param="F", atoms=at)
    else:
        for i in range(3):
            at.append(mol[int(list_splited_str[i + 1]) - 1])
        constr = constrain(type="A", atoms=at, angle=float(list_splited_str[-1]))

    return constr


def parcer_D(string, mol):
    at = []
    list_splited_str = string_ready_list(string)

    if list_splited_str[-1] == "F":
        for i in range(4):
            at.append(mol[int(list_splited_str[i + 1]) - 1])
        constr = constrain(type="D", param="F", atoms=at)
    else:
        for i in range(4):
            at.append(mol[int(list_splited_str[i + 1]) - 1])
        constr = constrain(type="D", atoms=at, angle=float(list_splited_str[-1]))

    return constr


class atom:
    def __init__(self, charge, position=[], number_in_mol=0):
        self.charge = charge
        self.pos = np.array(position, dtype=float)
        self.number_in_mol = number_in_mol

    def __str__(self):
        string = ""
        string = string + "N: " + str(self.number_in_mol) + " CHARGE: " + str(self.charge) + " POS: "

        for i in self.pos:
            string = string + str(i) + "; "
        return string


class B_func:
    def __init__(self, const, point):
        self.point = point
        self.con = const

    def get_value(self):
        vec1 = self.point[3 * self.con.n_atoms[0].number_in_mol - 3: 3 * self.con.n_atoms[0].number_in_mol]
        vec2 = self.point[3 * self.con.n_atoms[1].number_in_mol - 3: 3 * self.con.n_atoms[1].number_in_mol]

        return (vec1 - vec2).dot(vec1 - vec2) - self.con.length ** 2

    def get_grad(self):
        grad = np.zeros((3, int(len(self.point) / 3)))
        grad = grad.reshape(int(len(self.point) / 3), 3)
        point = self.point.reshape(int(len(self.point) / 3), 3)

        vec1 = point[self.con.n_atoms[0].number_in_mol - 1]
        vec2 = point[self.con.n_atoms[1].number_in_mol - 1]

        grad[self.con.n_atoms[0].number_in_mol - 1] = 2 * (vec1 - vec2)
        grad[self.con.n_atoms[1].number_in_mol - 1] = 2 * (vec2 - vec1)

        return grad.reshape(len(self.point), 1)

    def get_hess(self):
        hess = np.zeros((len(self.point), len(self.point)))
        hess[3 * self.con.n_atoms[0].number_in_mol - 3: 3 * self.con.n_atoms[0].number_in_mol,
        3 * self.con.n_atoms[1].number_in_mol - 3: 3 * self.con.n_atoms[1].number_in_mol] = -2
        hess[3 * self.con.n_atoms[1].number_in_mol - 3: 3 * self.con.n_atoms[1].number_in_mol,
        3 * self.con.n_atoms[0].number_in_mol - 3: 3 * self.con.n_atoms[0].number_in_mol] = -2
        hess[3 * self.con.n_atoms[1].number_in_mol - 3: 3 * self.con.n_atoms[1].number_in_mol,
        3 * self.con.n_atoms[1].number_in_mol - 3: 3 * self.con.n_atoms[1].number_in_mol] = -2
        hess[3 * self.con.n_atoms[0].number_in_mol - 3: 3 * self.con.n_atoms[0].number_in_mol,
        3 * self.con.n_atoms[0].number_in_mol - 3: 3 * self.con.n_atoms[0].number_in_mol] = -2
        for i in range(3):
            hess[3 * self.con.n_atoms[0].number_in_mol - (3 - i), 3 * self.con.n_atoms[0].number_in_mol - (3 - i)] = 2
            hess[3 * self.con.n_atoms[1].number_in_mol - (3 - i), 3 * self.con.n_atoms[1].number_in_mol - (3 - i)] = 2
        return hess


class A_func:
    def __init__(self, const, point):
        self.point = point
        self.con = const

    def get_value(self):
        vec1 = self.point[3 * self.con.n_atoms[0].number_in_mol - 3: 3 * self.con.n_atoms[0].number_in_mol]
        vec2 = self.point[3 * self.con.n_atoms[1].number_in_mol - 3: 3 * self.con.n_atoms[1].number_in_mol]
        vec3 = self.point[3 * self.con.n_atoms[2].number_in_mol - 3: 3 * self.con.n_atoms[2].number_in_mol]

        tmp1 = vec1 - vec2
        tmp2 = vec3 - vec2
        return tmp1.dot(tmp2) / (np.linalg.norm(tmp1) * np.linalg.norm(tmp2))

    def get_grad(self):
        return 1

    def get_hess(self):
        return 1


class D_func:
    def __init__(self, const, point):
        self.point = point
        self.con = const

    def get_value(self):
        vec1 = self.point[3 * self.con.n_atoms[0].number_in_mol - 3: 3 * self.con.n_atoms[0].number_in_mol]
        vec2 = self.point[3 * self.con.n_atoms[1].number_in_mol - 3: 3 * self.con.n_atoms[1].number_in_mol]
        vec3 = self.point[3 * self.con.n_atoms[2].number_in_mol - 3: 3 * self.con.n_atoms[2].number_in_mol]
        vec4 = self.point[3 * self.con.n_atoms[3].number_in_mol - 3: 3 * self.con.n_atoms[3].number_in_mol]
        tmp1 = vec1 - vec2
        tmp2 = vec3 - vec2
        tmp3 = vec3 - vec4

        angle = np.arctan2(np.cross(np.cross(tmp1, tmp2), np.cross(tmp2, tmp3)).dot(tmp2 / np.linalg.norm(tmp2)),
                           np.cross(tmp1, tmp2).dot(np.cross(tmp2, tmp3)))

        return angle

    def get_grad(self):
        return 1

    def get_hess(self):
        return 1

class constrain:
    constr_type = ""
    constr_param = ""
    n_atoms = []
    length = .0
    angle = .0

    def __init__(self, type="", param="", atoms=[], len=.0, angle=.0):
        self.constr_type = type
        self.constr_param = param
        self.n_atoms = atoms
        self.length = len
        self.angle = angle

        if self.constr_param == "F":
            if self.constr_type == "B":
                self.length = calc_B(self.n_atoms[0].pos, self.n_atoms[1].pos)
            elif self.constr_type == "A":
                self.angle = calc_A(self.n_atoms[0].pos, self.n_atoms[1].pos, self.n_atoms[2].pos)
            elif self.constr_type == "D":
                self.angle = calc_D(self.n_atoms[0].pos, self.n_atoms[1].pos, self.n_atoms[2].pos, self.n_atoms[3].pos)

    def __str__(self):
        string = "TYPE: "
        if self.constr_type == "B":
            string = string + "LENGTH\n"
        elif self.constr_type == "A":
            string = string + "VALENCE ANGLE\n"
        elif self.constr_type == "D":
            string = string + "DIHEDRAL ANGLE\n"

        for i in self.n_atoms:
            string = string + str(i) + " \n"

        if self.constr_param == "F":
            string = string + "PARAM FROZEN -- "
            if self.constr_type == "B":
                string = string + "LENGTH: " + str(self.length)
            elif self.constr_type == "A":
                string = string + "VALENCE ANGLE: " + str(self.angle)
            elif self.constr_type == "D":
                string = string + "DIHEDRAL ANGLE: " + str(self.angle)
        else:
            string = string + "PARAM SET -- "
            if self.constr_type == "B":
                string = string + "LENGTH: " + str(self.length)
            elif self.constr_type == "A":
                string = string + "VALENCE ANGLE: " + str(self.angle)
            elif self.constr_type == "D":
                string = string + "DIHEDRAL ANGLE: " + str(self.angle)
        return string


class constrains:
    CONSTR_LIST = []
    atoms = []

    def __init__(self, cnstr_path_file, atoms=[]):
        self.file_path = cnstr_path_file
        self.atoms = atoms

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


            for line in list_lines:
                line = line.strip(' ')
                line = line.strip("\n")

                if line[0] == 'B':
                    line_B.append(line)
                elif line[0] == 'A':
                    line_A.append(line)
                elif line[0] == 'D':
                    line_D.append(line)

            for it in line_B:
                self.CONSTR_LIST.append(pacer_B(it, self.atoms))
            for it in line_A:
                self.CONSTR_LIST.append(parcer_A(it, self.atoms))
            for it in line_D:
                self.CONSTR_LIST.append(parcer_D(it, self.atoms))

            file.close()

    def get_constr_val(self, point):
        list_val = []
        nLambda = len(self.CONSTR_LIST)

        for i in self.CONSTR_LIST:
            if i.constr_type == "B":
                con = B_func(i, point)
                list_val.append(con.get_value())
            if i.constr_type == "A":
                con = A_func(i, point)
                list_val.append(con.get_value())
            if i.constr_type == "D":
                con = D_func(i, point)
                list_val.append(con.get_value())

        vect = np.array(list_val)
        vect = vect.reshape(nLambda, 1)

        return vect

    def get_constr_grad(self, point):
        list_grad = []
        nInter = len(point)
        nLambda = len(self.CONSTR_LIST)
        matr_grad = np.zeros((nInter, nLambda))

        for i in self.CONSTR_LIST:
            if i.constr_type == "B":
                con = B_func(i, point)
                list_grad.append(con.get_grad())
            if i.constr_type == "A":
                con = A_func(i, point)
                list_grad.append(con.get_grad())
            if i.constr_type == "D":
                con = D_func(i, point)
                list_grad.append(con.get_grad())

        for i in range(nLambda):
            matr_grad[:, [i]] = list_grad[i]

        return matr_grad

    def get_constr_hess(self, point):
        list_hess = []

        for i in self.CONSTR_LIST:
            if i.constr_type == "B":
                con = B_func(i, point)
                list_hess.append(con.get_hess())
            if i.constr_type == "A":
                con = A_func(i, point)
                list_hess.append(con.get_hess())
            if i.constr_type == "D":
                con = D_func(i, point)
                list_hess.append(con.get_hess())

        return list_hess
