from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles
import numpy as np
from rdkit import Chem

fun_group_string = """Acetylide
C#C
Carbonyl
[$([CX3]=[OX1]),$([CX3+]-[OX1-])]
Acyl Halide
[CX3](=[OX1])[F,Cl,Br,I]
Aldehyde
[CX3H1](=O)[#6]
Anhydride
[CX3](=[OX1])[OX2][CX3](=[OX1])
Amide
[NX3][CX3](=[OX1])[#6]
Amidinium
[NX3][CX3]=[NX3+]
Carbamate
[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]
Carbamic ester
[NX3][CX3](=[OX1])[OX2H0]
Carboxyl
[CX3](=O)[OX2H1]
Cyanamide
[NX3][CX2]#[NX1]
Ester
[#6][CX3](=O)[OX2H0][#6]
Ketone
[#6][CX3](=O)[#6]
Ether
[OD2]([#6])[#6]
Fulminate
C=NO
primary amine
[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]
secondary amine
[NX3;H;!$(NC=[!#6]);!$(NC#[!#6])][#6]
Enamine
[NX3][$(C=C),$(cc)]
Azide
[$(N=[N+]=[N-]),$([N-][N+]#N)]
Azo Nitrogen
[NX2]=N
Azoxy Nitrogen
[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]
Diazo Nitrogen
[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]
Azole
[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]
Substituted imine
[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]
Substituted or un-substituted imine
[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]
Iminium
[NX3+]=[CX3]
Nitrate
[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]
Nitrate ester
ON(O)O
Nitrile
[NX1]#[CX2]
Isonitrile
[CX1-]#[NX2+]
Nitro
[$([NX3](=O)=O),$([NX3+](=O)[O-]),$(N(=O)=O)][!#8]
Nitrite
[OX2][NX2]=[OX1]
nitroso
[NX2]=[OX1]
N-Oxide
[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]
Hydroxyl
[OX2H]
Enol
[OX2H][#6X3]=[#6]
Phenol
[OX2H][cX3]:[c]
Peroxide
[OX2,OX1-][OX2,OX1-]
C-F
CF
N-F2
N(F)F
N-F
NF
pyyro ring
[nX2r5]
benzene
[cR]1[cR][cR][cR][cR][cR]1
fused benzenes
c12ccccc1cccc2
pyridine
c1ccncc1
diazine
[$(c1cncnc1),$(c1nccnc1),$(c1cnncc1)]
"""

def functional_group_featurizer(mol_list, drop_empty_features=True):
    """

    Arguments:
        mol_list :: a list of rdkit mol objects

    Returns:
         (X, fun_names) :: as a NumPy array and a list

    """

    lines_to_process = fun_group_string.splitlines()

    num_fun_groups = len(lines_to_process)//2

    num_mols = len(mol_list)

    fun_names = []
    fun_group_mols = []

    for i in range(num_fun_groups):
        fun_names += [lines_to_process[2*i]]
        fun_group_mols += [MolFromSmarts(lines_to_process[2*i+1])]

    X = np.zeros((num_mols, num_fun_groups))

    for (m, mol) in enumerate(mol_list):
        for i in range(num_fun_groups):
            X[m,i] = len(mol.GetSubstructMatches(fun_group_mols[i]))

    if (drop_empty_features):

        fun_names = np.array(fun_names).reshape(1,-1)

        cols_to_delete = []
        for i in range(num_fun_groups):
            if (sum(X[:,i]) == 0):
                cols_to_delete += [i]

        #truncate
        X = np.delete(X, cols_to_delete, 1)
        fun_names = np.delete(fun_names, cols_to_delete, 1)

        fun_names = list(fun_names[0])

    return fun_names, X
