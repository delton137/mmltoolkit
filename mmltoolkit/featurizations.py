import numpy as np
import copy
from rdkit import Chem
from rdkit.Chem.rdmolops import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix
from rdkit.Chem.Graphs import CharacteristicPolynomial


#----------------------------------------------------------------------------
def literal_bag_of_bonds(mol_list):
    '''
        Note: Bond types are labeled according convention where the atom of the left is alphabetically less than
        the atom on the right. For instance, 'C=O' and 'O=C' bonds are lumped together under 'C=O', and NOT 'O=C'.

    Args:
        mol_list : a single mol object or list/iterable containing the RDKit mol objects for all of the molecules.
    Returns:
        bond_types : a list of strings describing the bond types in the feature vector
        X_LBoB : a NumPy array containing the feature vectors of shape (num_mols, num_bond_types)
    '''

    if (isinstance(mol_list, list) == False):
        mol_list = [mol_list]

    empty_bond_dict = {}
    num_mols = len(mol_list)

    #first pass through to enumerate all bond types in all molecules
    for i, mol in enumerate(mol_list):
        bonds = mol.GetBonds()
        for bond in bonds:
            bond_start_atom = bond.GetBeginAtom().GetSymbol()
            bond_end_atom = bond.GetEndAtom().GetSymbol()
            bond_type = bond.GetSmarts(allBondsExplicit=True)
            bond_atoms = [bond_start_atom, bond_end_atom]
            if (bond_type == ''):
                bond_type = "-"
            bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
            try:
                empty_bond_dict[bond_string] = 0
            except KeyError:
                empty_base_bond_dict[bond_string] = 0

    #second pass through to construct X
    bond_types = list(empty_bond_dict.keys())
    num_bond_types = len(bond_types)

    X_LBoB = np.zeros([num_mols, num_bond_types])

    for i, mol in enumerate(mol_list):
        bonds = mol.GetBonds()
        bond_dict = copy.deepcopy(empty_bond_dict)
        for bond in bonds:
            bond_start_atom = bond.GetBeginAtom().GetSymbol()
            bond_end_atom = bond.GetEndAtom().GetSymbol()
            bond_type = bond.GetSmarts(allBondsExplicit=True)
            if (bond_type == ''):
                bond_type = "-"
            bond_atoms = [bond_start_atom, bond_end_atom]
            bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
            bond_dict[bond_string] += 1

        X_LBoB[i,:] = [bond_dict[bond_type] for bond_type in bond_types]

    return bond_types, X_LBoB


#----------------------------------------------------------------------------
def adjacency_matrix_eigenvalues(mol_list, useBO=False):

    eigenvalue_list = []
    max_length = 0


    for mol in mol_list:
        adj_matrix = GetAdjacencyMatrix(mol, useBO=useBO)
        evs = np.linalg.eigvals(adj_matrix)
        evs = np.real(evs)
        evs = sorted(evs, reverse=True) #sort
        eigenvalue_list += [evs]
        length = len(evs)
        if (length > max_length):
            max_length = length

    #zero padding
    for i in range(len(eigenvalue_list)):
        pad_width = max_length - len(eigenvalue_list[i])
        eigenvalue_list[i] += [0]*pad_width

    return np.array(eigenvalue_list)

#----------------------------------------------------------------------------
def distance_matrix_eigenvalues(mol_list, invert=False):
    eigenvalue_list = []
    max_length = 0

    for mol in mol_list:
        matrix = GetDistanceMatrix(mol)
        if (invert):
            matrix = np.reciprocal(matrix)
            matrix = np.nan_to_num(matrix)

        evs = np.linalg.eigvals(matrix)
        evs = np.real(evs)
        evs = sorted(evs, reverse=True) #sort (should be default Numpy sbehaviour)
        eigenvalue_list += [evs]
        length = len(evs)
        if (length > max_length):
            max_length = length

    #zero padding
    for i in range(len(eigenvalue_list)):
        pad_width = max_length - len(eigenvalue_list[i])
        eigenvalue_list[i] += [0]*pad_width

    return np.array(eigenvalue_list)

#----------------------------------------------------------------------------
def characteristic_poly(mol_list, useBO=False):

    eigenvalue_list = []
    max_length = 0

    for mol in mol_list:
        evs = CharacteristicPolynomial(mol, GetAdjacencyMatrix(mol, useBO=True))
        #evs = sorted(evs, reverse=True) #sort
        eigenvalue_list += [list(evs)]
        length = len(evs)
        if (length > max_length):
            max_length = length

    #zero padding
    for i in range(len(eigenvalue_list)):
        pad_width = max_length - len(eigenvalue_list[i])
        eigenvalue_list[i] += [0]*pad_width

    return np.array(eigenvalue_list)
