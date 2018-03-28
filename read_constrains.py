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
        print("NOT FOUND: File with constrains")
    else:
        file = open(file_path, 'r')

        for line in file:
            line = string_ready_list(line)

            charge = 0
            comp_pos = []

            for i in range(len(line)):
                if i == 0:
                    charge = int(line[i])
                else:
                    comp_pos.append(float(line[i]))
            list_atoms.append(atom(charge, comp_pos))
        return list_atoms


def rad_to_grad(angle):
    return angle / np.pi * 180.0


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
    tmp3 = atom_3 - atom_4

    angle = np.arctan2(np.cross(np.cross(tmp1, tmp2), np.cross(tmp2, tmp3)).dot(tmp2 / np.linalg.norm(tmp2, ord=2)),
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
        at.append(mol[int(list_splited_str[1]) - 1])
        at.append(mol[int(list_splited_str[2]) - 1])
        at.append(mol[int(list_splited_str[3]) - 1])
        constr = constrain(type="A", param="F", atoms=at)
    else:
        at.append(mol[int(list_splited_str[1]) - 1])
        at.append(mol[int(list_splited_str[2]) - 1])
        at.append(mol[int(list_splited_str[3]) - 1])
        constr = constrain(type="A", atoms=at, angle=float(list_splited_str[-1]))

    return constr


def parcer_D(string, mol):
    at = []
    list_splited_str = string_ready_list(string)

    if list_splited_str[-1] == "F":
        at.append(mol[int(list_splited_str[1]) - 1])
        at.append(mol[int(list_splited_str[2]) - 1])
        at.append(mol[int(list_splited_str[3]) - 1])
        at.append(mol[int(list_splited_str[4]) - 1])
        constr = constrain(type="D", param="F", atoms=at)
    else:
        at.append(mol[int(list_splited_str[1]) - 1])
        at.append(mol[int(list_splited_str[2]) - 1])
        at.append(mol[int(list_splited_str[3]) - 1])
        at.append(mol[int(list_splited_str[4]) - 1])
        constr = constrain(type="D", atoms=at, angle=float(list_splited_str[-1]))

    return constr


class atom:
    def __init__(self, charge, position=[]):
        self.charge = charge
        self.pos = np.array(position, dtype=float)

    def __str__(self):
        string = ""
        string = string + "CHARGE: " + str(self.charge) + " POS: "

        for i in self.pos:
            string = string + str(i) + "; "
        return string


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
    COUNT_OF_CNSTR = 0
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

            self.COUNT_OF_CNSTR = int(string_ready_list(list_lines[0])[0])

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
