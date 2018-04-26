"""
    descriptors based on stoichiometry and functional groups
    Daniel C. Elton
    License : MIT
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, AddHs
import numpy as np
from rdkit.Chem.Descriptors import _descList
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator







def get_num_atom(mol, atomic_number):
    '''returns the number of atoms of particular atomic number'''
    num = 0
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if (atom_num == atomic_number):
            num += 1
    return num


def return_atom_nums(mol):
    ''' returns a vector with the number of C,N,H,O,F atoms'''
    n_O = get_num_atom(mol,8)
    n_C = get_num_atom(mol,6)
    n_N = get_num_atom(mol,7)
    n_H = get_num_atom(mol,1)
    n_F = get_num_atom(mol,9)
    return [n_O, n_C, n_N, n_H, n_F]


def get_neigh_dict(atom):
    '''returns a dictionary with the number of neighbors for a given atom of types C,N,H,O,F'''
    neighs = {'N':0,'C':0,'O':0,'H':0,'F':0, '*':0}
    for atom in atom.GetNeighbors():
        neighs[atom.GetSymbol()] += 1
    return neighs


def get_num_with_neighs(mol, central_atom, target_dict):
    '''returns how many atoms of a particular type have a particular configuration of neighbora'''
    target_num = 0
    for key in list(target_dict.keys()):
        target_num += target_dict[key]

    num = 0
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == central_atom):
            target = True
            nbs = get_neigh_dict(atom)
            for key in list(target_dict.keys()):
                if (nbs[key] != target_dict[key]):
                    target = False
                    break

            n_nbs = len(atom.GetNeighbors())
            if (target_num != n_nbs):
                target = False

            if (target):
                num +=1

    return num


def oxygen_balance_1600(mol):
    '''returns the OB_16000 descriptor'''
    n_O = get_num_atom(mol, 8)
    n_C = get_num_atom(mol, 6)
    n_H = get_num_atom(mol, 1)
    mol_weight = Descriptors.ExactMolWt(mol)
    return 1600*(n_O - 2*n_C - n_H/2)/mol_weight


def oxygen_balance_100(mol):
    '''returns the OB_100 descriptor'''
    n_O = get_num_atom(mol, 8)
    n_C = get_num_atom(mol, 6)
    n_H = get_num_atom(mol, 1)
    n_atoms = mol.GetNumAtoms()
    return 100*(n_O - 2*n_C - n_H/2)/n_atoms

def return_combined_nums(mol):
    return custom_descriptor_set(mol)

def custom_descriptor_set(mol):
    n_a = mol.GetNumAtoms()
    n_C = get_num_atom(mol, 'C')
    n_N = get_num_atom(mol, 'N')
    n_O = get_num_atom(mol, 'O')
    n_H = get_num_atom(mol, 'H')
    n_F = get_num_atom(mol, 'F')
    n_O1 = get_num_with_neighs(mol, 'O', {'N': 1})
    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})

    if (n_C == 0):
        NCratio = n_N/0.00001
    else:
        NCratio = n_N/n_C

    return [oxygen_balance_100(mol), n_C, n_N, n_O1, n_O2, n_O3, n_O4, n_H, n_F, NCratio,
                get_num_with_neighs(mol, 'N', {'O': 2, 'C': 1}) ,  #CNO2
                get_num_with_neighs(mol, 'N', {'O': 2, 'N': 1}) ,  #NNO2
                get_num_with_neighs(mol, 'N', {'O': 2})         ,  #ONO
                get_num_with_neighs(mol, 'N', {'O': 3})         ,  #ONO2
                get_num_with_neighs(mol, 'N', {'N': 1, 'C': 1}) ,  #CNN
                get_num_with_neighs(mol, 'N', {'N': 2})         ,  #NNN
                get_num_with_neighs(mol, 'N', {'C': 1,'O': 1})  ,  #CNO
                get_num_with_neighs(mol, 'N', {'C': 1,'H': 2})  ,  #CNH2
                get_num_with_neighs(mol, 'N', {'C': 2,'O': 1})  ,  #CN(O)C
                get_num_with_neighs(mol, 'F', {'C': 1})         ,  #CF
                get_num_with_neighs(mol, 'N', {'C': 1, 'N':2})     #CNF
                #n_C/n_a, n_N/n_a, n_O/n_a, n_H/n_a
           ]

def modified_oxy_balance(mol):
    '''returns an OB_100 descriptor with modified oxygen types
        ref: A.  R.  Martin  and  H.  J.  Yallop,  Trans.  Faraday  Soc.
                54,  257(1958), URLhttp://dx.doi.org/10.1039/TF9585400257.
    '''
    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})
    n_atoms = mol.GetNumAtoms()

    OB = oxygen_balance_100(mol)

    correction = 100*(1.0*n_O2 + 1.8*n_O2 + 2.2*n_O3)/n_atoms

    #if (OB > 0 ):
    #    mod_OB = OB - correction

    #if (OB <= 0 ):
    #    mod_OB = OB + correction

    return OB - correction

def custom_oxy_balance(mol, w1=1, w2=0.3, w3=-0.6, w4=-1.8, w5=-1.9, w6=-0.5):
    '''returns output of a linear model based on atom types obtained by Martin \& Yallop
        ref: A.  R.  Martin  and  H.  J.  Yallop,  Trans.  Faraday  Soc.
                54,  257(1958), URLhttp://dx.doi.org/10.1039/TF9585400257.
    '''
    n_C = get_num_atom(mol,6)
    n_N = get_num_atom(mol,7)
    n_H = get_num_atom(mol,1)
    n_O1 = get_num_with_neighs(mol, 'O', {'N': 1})
    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})
    mol_weight = Descriptors.ExactMolWt(mol)
    return w1*n_O1 + w2*n_O2 + w3*n_O3 + w4*n_O4 + w5*n_C + w6*n_H


def return_atom_nums_modified_OB(mol):
    '''returns number of different atoms, including modified oxygen types
        ref: A.  R.  Martin  and  H.  J.  Yallop,  Trans.  Faraday  Soc.
                54,  257(1958), URLhttp://dx.doi.org/10.1039/TF9585400257.
    '''
    n_C = get_num_atom(mol, 6)
    n_N = get_num_atom(mol, 7)
    n_H = get_num_atom(mol, 1)
    n_F = get_num_atom(mol, 9)
    n_O1 = get_num_with_neighs(mol, 'O', {'N': 1})
    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})
    return [n_C, n_N, n_H, n_O1, n_O2, n_O3, n_O4, n_F]



def RDKit_descriptor_featurizer(mol_list, descriptor_list=_descList, return_names=True):
    num_descriptors = len( descriptor_list)
    num_mols = len(mol_list)
    descriptor_function_names = [_descList[i][0] for i in range(num_descriptors)]

    mdc = MolecularDescriptorCalculator(simpleList=descriptor_function_names)

    X = np.zeros([num_mols, num_descriptors])

    for i in range(num_mols):
        X[i,:] = np.array(mdc.CalcDescriptors(mol_list[i]))

    descriptor_function_names = np.array(descriptor_function_names)

    #Drop descriptors that are zero for every molecule in the list
    cols_to_drop = []
    for i in range(num_descriptors):
        if (sum(X[:,i]) == 0):
            cols_to_drop += [i]

    X_truncated = np.delete(X, cols_to_drop, 1)
    descriptor_function_names_truncated = np.delete(descriptor_function_names, cols_to_drop, 0)

    descriptor_function_names = list(descriptor_function_names)

    if (return_names):
        return descriptor_function_names_truncated, X_truncated
    else:
        return X_truncated
