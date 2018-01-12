import numpy as np
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP #GetAvalonCountFP  #int vector version
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.model_selection import cross_val_score
import rdkit.DataStructs.cDataStructs
from rdkit.Chem import Descriptors

class fingerprint():
    def __init__(self, fp_fun, name):
        self.fp_fun = fp_fun
        self.name = name
        self.x = []

    def apply_fp(self, mols):
        for mol in mols:
            fp = self.fp_fun(mol)
            if isinstance(fp, tuple):
                fp = np.array(list(fp[0]))
            if isinstance(fp, rdkit.DataStructs.cDataStructs.ExplicitBitVect):
                fp = ExplicitBitVect_to_NumpyArray(fp)
            if isinstance(fp,rdkit.DataStructs.cDataStructs.IntSparseIntVect):
                fp = np.array(list(fp))

            self.x += [fp]

            if (str(type(self.x[0])) != "<class 'numpy.ndarray'>"):
                print("WARNING: type for ", self.name, "is ", type(self.x[0]))

        #Scale fingerprint to unit variance and zero mean
        #st = StandardScaler()
        #self.x = st.fit_transform(self.x)

def truncated_Estate_fp(mol_list):
    return np.array([FingerprintMol(mol)[0][6:37] for mol in mol_list])

def fp_Estate_ints(mol):
    return FingerprintMol(mol)[0][6:37]

def fp_Estate_reals(mol):
    return FingerprintMol(mol)[1][6:37]

def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

def fp_Estate_and_mw(mol):
    return np.append(FingerprintMol(mol)[0][6:37], Descriptors.MolWt(mol))


def make_fingerprints(mols, length = 1024, verbose=False):

    fp_list = [
         #fingerprint(lambda x : GetBPFingerprint(x, fpfn=GetHashedAtomPairFingerprintAsBitVect),
         #            "Physiochemical properties (1996)"), ##NOTE: takes a long time to compute
         fingerprint(lambda x : GetHashedAtomPairFingerprintAsBitVect(x, nBits = length),
                     "Atom pair (1985)"),
         fingerprint(lambda x : GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits = length),
                     "Topological Torsion (1987)"),
         fingerprint(lambda x : GetMorganFingerprintAsBitVect(x, 2, nBits = length),
                     "ECFPs/Morgan Circular (2010) "),
         fingerprint(fp_Estate_ints, "E-state (fixed length) (1995)"),
         #fingerprint(fp_Estate_and_mw, "E-state + MW weight (1995)"),
         #fingerprint(FingerprintMol, "E-state, index sum (1995)"),
         fingerprint(lambda x: GetAvalonFP(x, nBits=length),
                    "Avalon (2006)"),
         #fingerprint(lambda x: np.append(GetAvalonFP(x, nBits=length), Descriptors.MolWt(x)),
         #           "Avalon+mol. weight"),
         #fingerprint(lambda x: GetErGFingerprint(x), "ErG (2006)"),
         fingerprint(lambda x : RDKFingerprint(x, fpSize=length),
                     "RDKit topological (2006)")
    ]

    for fp in fp_list:
        if (verbose): print("doing", fp.name)

        fp.apply_fp(mols)

    return fp_list

#---------------------------------------------------------------
def test_fingerprints(fp_list, model, y, verbose = True):

    fingerprint_scores = {}

    for fp in fp_list:
        if verbose: print("doing ", fp.name)

        scores = cross_val_score(model, fp.x, y, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
        fingerprint_scores[fp.name] = -1*np.mean(scores)

    sorted_names = sorted(fingerprint_scores, key=fingerprint_scores.__getitem__, reverse=False)

    print("\\begin{tabular}{c c}")
    print("           name        &  avg abs error in CV (kJ/cc) \\\\")
    print("\\hline")
    for name in sorted_names:
        print("%30s & %5.3f \\\\" % (name, fingerprint_scores[name]))
    print("\\end{tabular}")
