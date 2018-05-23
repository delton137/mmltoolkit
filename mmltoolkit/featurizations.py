"""
Note : currently the Coulomb Matrix and Bag of Bonds functions take .xyz filenames as inputs
       in the future, they could be rewritten to work on RDKit mol objects, like the rest of the
       featurization functions.
"""
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.rdmolops import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix
from rdkit.Chem.Graphs import CharacteristicPolynomial
from rdkit.Chem.Descriptors import _descList
from collections import defaultdict
from .fingerprints import truncated_Estate_featurizer
from .descriptors import RDKit_descriptor_featurizer
from .functional_group_featurizer import functional_group_featurizer
atom_num_dict = {'C':6,'N':7,'O':8,'H':1,'F':9, 'Cl': 17, 'S': 16 }

#----------------------------------------------------------------------------
def bag_of_bonds(filename_list, verbose=False):
    """
    REF:
        Hansen, et al., The Journal of Physical Chemistry Letters 2015 6 (12), 2326-2331
        DOI: 10.1021/acs.jpclett.5b00831, URL: http://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00831
    Args:
        filename_list : a list containing strings for all of the .xyz input filenames
    Returns:
        feature_names : a (long) list of strings describing which bag each element of the feature vector is part of
        X_LBoB : a NumPy array containing the feature vectors of shape (num_mols, num_bond_types)
    """
    import copy

    num_mols = len(filename_list)

    #------- initialize empty dictionary for storing each bag as a list -----
    atom_types = ['C', 'N', 'O', 'F', 'H']
    num_atom_types = len(atom_types)

    empty_BoB_dict = {}
    for atom_type in atom_types:
        empty_BoB_dict[atom_type] = [] #initialize empty list

    for i in range(num_atom_types):
        for j in range(i,num_atom_types):
            empty_BoB_dict[atom_types[i]+atom_types[j]] = [] #initialize empty list

    #------------- fill dicts in dict list ------------------------------------
    BoB_dict_list = []
    if (verbose): print("creating intial BoBs")

    for m, filename in enumerate(filename_list):
        xyzfile = open(filename, 'r')
        num_atoms_file = int(xyzfile.readline())
        xyzfile.close()
        Cmat = np.zeros((num_atoms_file,num_atoms_file))
        chargearray = np.zeros((num_atoms_file, 1))
        xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
        atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
        atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
        chargearray = [atom_num_dict[symbol] for symbol in atom_symbols]

        BoB_dict = copy.deepcopy(empty_BoB_dict)

        #------- populate BoB dict ------------------------------------------
        for i in range(num_atoms_file):
            for j in range(i, num_atoms_file):
                if i == j:
                    BoB_dict[atom_symbols[i]] += [0.5*chargearray[i]**2.4] #concactenate to list
                else:
                    dict_key = atom_symbols[i]+atom_symbols[j]
                    dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                    CM_term = chargearray[i]*chargearray[j]/dist
                    try:
                        BoB_dict[dict_key] += [CM_term] #concactenate to list
                    except KeyError:
                        dict_key = atom_symbols[j]+atom_symbols[i]
                        BoB_dict[dict_key] += [CM_term] #concactenate to list

        BoB_dict_list += [BoB_dict]


    #------- tricky processing stage - zero pad all bags so they all have the same length
    #------- and then cocatenate all bags into a feature vector for each molecule

    #For each key in the dict, zero pad the bags, and do this for all molecules
    #also sum these up to get the total length of the final feature vector
    feature_vect_length = 0

    if (verbose): print("finding max length of each bag and padding")

    for key in BoB_dict_list[0].keys():
        max_length = 0
        #find max bag length
        for i in range(num_mols):
            length = len(BoB_dict_list[i][key])
            if (length > max_length):
                max_length = length

        if (verbose): print("max length of ", key, "is", max_length)
        #zero pad each bag
        for i in range(num_mols):
            pad_width = max_length - len(BoB_dict_list[i][key])
            BoB_dict_list[i][key] = BoB_dict_list[i][key]+[0]*pad_width
        feature_vect_length += max_length

    #initialize Numpy feature vector array
    X_BoB = np.zeros((num_mols, feature_vect_length))

    #concatenation of all bags
    if (verbose): print("concatenating the bags")

    for m in range(num_mols):
        featvec = []
        for key in BoB_dict_list[m].keys():
            featvec += sorted(BoB_dict_list[m][key], reverse=True) #Sort (finally)
        X_BoB[m,:] = np.array(featvec)

    #concatenate feature names
    feature_names = []
    for key in BoB_dict_list[0].keys():
        for element in BoB_dict_list[0][key]:
            feature_names += [key]

    return feature_names, X_BoB


#----------------------------------------------------------------------------
def summed_bag_of_bonds(filename):
    """
        Based on   Hansen, et al., The Journal of Physical Chemistry Letters 2015 6 (12), 2326-2331
        DOI: 10.1021/acs.jpclett.5b00831, URL: http://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00831
        However, the Coulomb matrix terms for each atom pair (C-C, C-N, C-O, etc) are **summed** together.
        The diagonal terms of the Coulomb matrix are concatenated with the resulting vector.
        So the resulting feature vector for each molecule is a vector of length
        (num_atom_pair_types + num_atom_types). This is different than the original BoB, which maintains each
        CM entry in the feature vector.
    Args:
        filename : (string) the .xyz input filename for the molecule
    Returns:
        (feature_names, BoB_list) as lists
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_num_dict[symbol] for symbol in atom_symbols]

    #------- initialize dictionary for storing each bag ---------
    atom_types = ['C', 'N', 'O', 'F', 'H']
    num_atom_types = len(atom_types)

    BoB_dict = {}
    for atom_type in atom_types:
        BoB_dict[atom_type] = 0

    for i in range(num_atom_types):
        for j in range(i,num_atom_types):
            BoB_dict[atom_types[i]+atom_types[j]] = 0

    #------- populate BoB dict -----------------------------------
    for i in range(num_atoms_file):
        for j in range(i, num_atoms_file):
            if i == j:
                BoB_dict[atom_symbols[i]] += 0.5*chargearray[i]**2.4
            else:
                dict_key = atom_symbols[i]+atom_symbols[j]
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                CM_term = chargearray[i]*chargearray[j]/dist
                try:
                    BoB_dict[dict_key] += CM_term
                except KeyError:
                    dict_key = atom_symbols[j]+atom_symbols[i]
                    BoB_dict[dict_key] += CM_term

    #------- process into list -------------------------------------
    feature_names = list(BoB_dict.keys())
    BoB_list = [BoB_dict[feature] for feature in feature_names]

    return feature_names, BoB_list

#----------------------------------------------------------------------------
def coulombmat_and_eigenvalues_as_vec(filename, padded_size, sort=True):
    """
    returns Coulomb matrix and **sorted** Coulomb matrix eigenvalues
    Args:
        filename : (string) the .xyz input filename for the molecule
        padded_size : the number of atoms in the biggest molecule to be considered (same as padded eigenvalue vector length)
    Returns:
        (Eigenvalues vector, Coulomb matrix vector) as Numpy arrays
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_num_dict[symbol] for symbol in atom_symbols]

    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if i == j:
                Cmat[i,j]=0.5*chargearray[i]**2.4   # Diagonal term described by Potential energy of isolated atom
            else:
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                Cmat[i,j]=chargearray[i]*chargearray[j]/dist   #Pair-wise repulsion

    Cmat_eigenvalues = np.linalg.eigvals(Cmat)
    #print(Cmat_eigenvalues)
    if (sort): Cmat_eigenvalues = sorted(Cmat_eigenvalues, reverse=True) #sort

    Cmat_as_vec = []
    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if (j>=i):
                Cmat_as_vec += [Cmat[i,j]]

    pad_width = (padded_size**2 - padded_size)//2 + padded_size - ((num_atoms_file**2 - num_atoms_file)//2 + num_atoms_file)
    Cmat_as_vec = Cmat_as_vec + [0]*pad_width

    Cmat_as_vec = np.array(Cmat_as_vec)

    pad_width = padded_size - num_atoms_file
    Cmat_eigenvalues = np.pad(Cmat_eigenvalues, ((0, pad_width)), mode='constant')

    return Cmat_eigenvalues, Cmat_as_vec


#----------------------------------------------------------------------------
def sum_over_bonds_single_mol(mol, bond_types):
    bonds = mol.GetBonds()
    bond_dict = defaultdict(lambda : 0)

    for bond in bonds:
        bond_start_atom = bond.GetBeginAtom().GetSymbol()
        bond_end_atom = bond.GetEndAtom().GetSymbol()
        bond_type = bond.GetSmarts(allBondsExplicit=True)
        if (bond_type == ''):
            bond_type = "-"
        bond_atoms = [bond_start_atom, bond_end_atom]
        bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
        bond_dict[bond_string] += 1

    X_LBoB = [bond_dict[bond_type] for bond_type in bond_types]

    return np.array(X_LBoB).astype('float64')


#----------------------------------------------------------------------------
def literal_bag_of_bonds(mol_list, predefined_bond_types=[]):
    return sum_over_bonds(mol_list, predefined_bond_types=predefined_bond_types)

#----------------------------------------------------------------------------
def sum_over_bonds(mol_list, predefined_bond_types=[]):
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

    if (len(predefined_bond_types) == 0 ):
        #first pass through to enumerate all bond types in all molecules and set them equal to zero in the dict
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
    else:
        for bond_string in predefined_bond_types:
            empty_bond_dict[bond_string] = 0

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
            #skip dummy atoms
            if (bond_start_atom=='*' or bond_end_atom=='*'):
                pass
            else:
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

#----------------------------------------------------------------------------
def CDS_featurizer(mol_list, return_names=True):
    from .descriptors import custom_descriptor_set
    X_CDS = []
    for mol in mol_list:
        X_CDS += [custom_descriptor_set(mol)]

    X = np.array(X_CDS)

    CDS_names = ['OB$_{100}$', 'n_C', 'n_N', 'n_NO', 'n_COH', 'n_NOC', 'n_CO', 'n_H', 'n_F', 'n_N/n_C', 'n_CNO2',
    'n$_{\\ff{NNO}_2}$', 'n$_{\\ff{ONO}}$', 'n$_{\\ff{ONO}_2}$', 'n_$\\ff{CNN}$', 'n_$\\ff{NNN}$', 'n_CNO', 'n_CNH2', 'n_CN(O)C', 'n_CF', 'n_CNF']

    if (return_names):
        return CDS_names, X
    else:
        return X

#----------------------------------------------------------------------------
def Estate_CDS_LBoB_featurizer(mol_list, predefined_bond_types=[], return_names=True):

    if (isinstance(mol_list, list) == False):
        mol_list = [mol_list]

    names_Estate, X_Estate = truncated_Estate_featurizer(mol_list, return_names=True )
    names_CDS, X_CDS = CDS_featurizer(mol_list, return_names=True)
    names_LBoB, X_LBoB = literal_bag_of_bonds(mol_list, predefined_bond_types=predefined_bond_types)

    X_combined = np.concatenate((X_Estate, X_CDS, X_LBoB), axis=1)
    X_scaled = StandardScaler().fit_transform(X_combined)

    names_all = list(names_Estate)+list(names_CDS)+list(names_LBoB)

    if (return_names):
        return names_all, X_scaled
    else:
        return X_scaled

#----------------------------------------------------------------------------
def Estate_CDS_LBoB_fungroup_featurizer(mol_list, predefined_bond_types=[], return_names=True, verbose=False):
    names_Estate, X_Estate = truncated_Estate_featurizer(mol_list, return_names=True )
    names_CDS, X_CDS = CDS_featurizer(mol_list, return_names=True)
    names_LBoB, X_LBoB = literal_bag_of_bonds(mol_list, predefined_bond_types=predefined_bond_types)
    names_fun, X_fun = functional_group_featurizer(mol_list)

    X_combined = np.concatenate((X_Estate, X_CDS, X_LBoB, X_fun), axis=1)
    X_scaled = StandardScaler().fit_transform(X_combined)

    names_all = list(names_Estate)+list(names_CDS)+list(names_LBoB)+list(names_fun)

    if (return_names):
        return names_all, X_scaled
    else:
        return X_scaled

#----------------------------------------------------------------------------
def all_descriptors_combined(mol_list, predefined_bond_types=[], rdkit_descriptor_list=_descList, return_names = True, verbose=False):
    names_RDkit, X_RDKit = RDKit_descriptor_featurizer(mol_list, descriptor_list=rdkit_descriptor_list)
    names_LBoB, X_LBoB = literal_bag_of_bonds(mol_list, predefined_bond_types=predefined_bond_types)
    names_CDS, X_CDS = CDS_featurizer(mol_list, return_names=True)
    names_Estate, X_Estate = truncated_Estate_featurizer(mol_list, return_names=True )
    names_fun, X_fun = functional_group_featurizer(mol_list)

    if verbose: print("number of RDKit descriptors used : %i" % (len(names_RDkit)))

    X_combined = np.concatenate((X_RDKit, X_CDS, X_LBoB, X_Estate, X_fun), axis=1)
    X_scaled = StandardScaler().fit_transform(X_combined)

    names_all = list(names_RDkit)+list(names_LBoB)+list(names_CDS)+list(names_Estate)+list(names_fun)

    if (return_names):
        return names_all, X_scaled
    else:
        return X_scaled
