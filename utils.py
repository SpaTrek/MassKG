from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
import numpy as np
import pandas as pd
import re
import itertools
from rdkit.Chem.Draw import IPythonConsole
from matplotlib import pyplot as plt
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.drawOptions.addBondIndices = True
IPythonConsole.molSize = 500,300



def draw_molb(got_ms2,mollist,savepath,mode):
    """
    Parameters
    ----------
    gms : TYPE
        DESCRIPTION.
    mollist : TYPE
        DESCRIPTION.
    savepath : TYPE
    mode : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from rdkit.Chem.Draw import rdMolDraw2D
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    gms['bonds'] = gms['bonds'].astype(str)
    for i,test_mol in enumerate(mollist):
        Chem.Kekulize(test_mol,clearAromaticFlags=True)
        gms_ = gms[gms['mid']==i]
        good_bonds = [eval(b) for b in gms_['bonds']]
        fw = list(gms_['mz'])
        if len(fw)>0: 
            for j in range(len(gms_)):
                if len(good_bonds[j])<1:
                    continue
                name = str(fw[j])
                d = rdMolDraw2D.MolDraw2DCairo(500, 400)
                hit_bonds = good_bonds[j]
                fmol = Chem.FragmentOnBonds(test_mol,hit_bonds) 
                try:
                    frag_mols = Chem.GetMolFrags(fmol,asMols=True) 
                except Exception as e:
                    print(e)
                    continue
                frag_smiles = [Chem.MolToSmiles(f) for f in frag_mols]
                bond_cols = {}
                for j, bd in enumerate(hit_bonds):
                    bond_cols[bd] = tuple(np.random.rand(3))
                rdMolDraw2D.PrepareAndDrawMolecule(d,test_mol,
                                                   highlightBonds=hit_bonds,
                                                   highlightBondColors=bond_cols)
                d.DrawMolecule(test_mol)
                d.FinishDrawing()
                d.WriteDrawingText(os.path.join(savepath,f'{name}.png'))        



