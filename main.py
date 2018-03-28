import read_constrains as cnstr_r

start_cinfig = cnstr_r.parser_xyz("/home/rusanov-vs/PycharmProjects/constr/in/struct.xyz")

constr = cnstr_r.constrains("/home/rusanov-vs/PycharmProjects/constr/in/cnstr", start_cinfig)

for i in constr.CONSTR_LIST:
    print(i)
