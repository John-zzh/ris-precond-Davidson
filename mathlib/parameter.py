#!/usr/bin/python
from arguments import args
Hartree_to_eV = 27.211386245988


'''a dictionary of chemical hardness, by mappig two lists:
   list of elements 1-94
   list of hardness for elements 1-94, floats, in Hartree
'''
elements = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca','Sc', 'Ti',
'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se',
'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr','Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
'Ag', 'Cd', 'In', 'Sn','Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce',
'Pr', 'Nd','Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Tl', 'Pb',
'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu']
hardness = [0.47259288,0.92203391,0.17452888,0.25700733,0.33949086,
0.42195412,0.50438193,0.58691863,0.66931351,0.75191607,0.17964105,
0.22157276,0.26348578,0.30539645,0.34734014,0.38924725,0.43115670,
0.47308269,0.17105469,0.20276244,0.21007322,0.21739647,0.22471039,
0.23201501,0.23933969,0.24665638,0.25398255,0.26128863,0.26859476,
0.27592565,0.30762999,0.33931580,0.37235985,0.40273549,0.43445776,
0.46611708,0.15585079,0.18649324,0.19356210,0.20063311,0.20770522,
0.21477254,0.22184614,0.22891872,0.23598621,0.24305612,0.25013018,
0.25719937,0.28784780,0.31848673,0.34912431,0.37976593,0.41040808,
0.44105777,0.05019332,0.06762570,0.08504445,0.10247736,0.11991105,
0.13732772,0.15476297,0.17218265,0.18961288,0.20704760,0.22446752,
0.24189645,0.25932503,0.27676094,0.29418231,0.31159587,0.32902274,
0.34592298,0.36388048,0.38130586,0.39877476,0.41614298,0.43364510,
0.45104014,0.46848986,0.48584550,0.12526730,0.14268677,0.16011615,
0.17755889,0.19497557,0.21240778,0.07263525,0.09422158,0.09920295,
0.10418621,0.14235633,0.16394294,0.18551941,0.22370139]
HARDNESS = dict(zip(elements,hardness))



common_elements = ['H', 'C', 'N', 'O', 'F', 'Si', 'S', 'Cl']
'''
GB Radii
Ghosh, Dulal C and coworkers
The wave mechanical evaluation of the absolute radii of atoms.
Journal of Molecular Structure: THEOCHEM 865, no. 1-3 (2008): 60-67.
'''

radii = [0.5292, 0.6513, 0.5428, 0.4652, 0.4071, 1.1477, 0.8739, 0.7808]
radii_au = [i*1.8897259885789 for i in radii]
#
RADII = dict(zip(common_elements,radii_au))
print(RADII)
RSH_F = [
'lc-b3lyp',
'wb97',
'wb97x',
'wb97x-d3',
'cam-b3lyp']
RSH_paramt = [
[0.53, 8.00, 4.50],
[0.61, 8.00, 4.41],
[0.56, 8.00, 4.58],
[0.51, 8.00, 4.51],
[0.38, 1.86, 0.90]]
RSH_F_paramt = dict(zip(RSH_F, RSH_paramt))

hybride_F = [
'b3lyp',
'tpssh',
'm05-2x',
'pbe0',
'm06',
'm06-2x',
'NA']
hybride_ax = [
0.2,
0.1,
0.56,
0.25,
0.27,
0.54,
1]
Func_ax = dict(zip(hybride_F, hybride_ax))

beta1 = 0.2
beta2 = 1.83
alpha1 = 1.42
alpha2 = 0.48

omega_dic = {}
omega_dic['pbe0'] = 0
omega_dic['wb97x'] = 0.3

RSH_omega = omega_dic[args.functional]
print('RSH_omega =', RSH_omega)




def gen_alpha_beta_ax(functional):

    ''' NA is for Hartree-Fork

        RSH functionals have specific a_x, beta, alpha values;
        hybride fucntionals have fixed alpha12 and beta12 values,
        with different a_x values, by which create beta, alpha
    '''
    if functional in RSH_F:
        a_x, beta, alpha = RSH_F_paramt[functional]
    elif functional in hybride_F:
        a_x = Func_ax[functional]
        beta = beta1 + beta2 * a_x
        alpha = alpha1 + alpha2 * a_x

    return a_x, beta, alpha
