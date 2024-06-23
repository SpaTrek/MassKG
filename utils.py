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
        不能有中文路径！！！.
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
        
def compare_structure(smiles1, smiles2, fp_type='Morgan', sim_type='Dice'):
    '''
    Task: 
        Compare structual similarity of two compound based on fingerprints.
    Parameters:
        smiles1: str, smiles of the compound 1
        smiles2: str, smiles of the compound 2
        fp_type: str, type of fingerprints
        sim_type: str, method for calculating similarity
    '''
    if fp_type == 'Morgan':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    elif fp_type == 'MorganWithFeature':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)
    elif fp_type == 'MACCS':
        getfp = lambda smi: Chem.MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi))
    elif fp_type == 'Topological':
        getfp = lambda smi: FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smi))
    elif fp_type == 'AtomPairs':
        getfp = lambda smi: Pairs.GetAtomPairFingerprint(Chem.MolFromSmiles(smi))
    
    try:
        fp1 = getfp(smiles1)
        fp2 = getfp(smiles2)
        if sim_type == 'Dice':
            sim_fp = DataStructs.DiceSimilarity(fp1, fp2)
        elif sim_type == 'Tanimoto':
            sim_fp = DataStructs.TanimotoSimilarity(fp1, fp2)
        elif sim_type == 'Cosine':
            sim_fp = DataStructs.CosineSimilarity(fp1, fp2)
        elif sim_type == 'Sokal':
            sim_fp = DataStructs.SokalSimilarity(fp1, fp2)
        elif sim_type == 'Russel':
            sim_fp = DataStructs.RusselSimilarity(fp1, fp2)
    except Exception as e:
        print(e)
        sim_fp = -1
    return sim_fp

def pop_list(l):
    out = []
    for x in l :
        
        if  len(out)<1:
            out.append(x)
        else:
            if not x in out:
                out.append(x)
    return out

def match_fragments(ms_table,refms2,mserror=20e-3):
    refms2 = refms2.copy()
    refms2['intensity'] = refms2['intensity']/refms2['intensity'].sum()
    if len(ms_table.shape) == 1:
        mslist = ms_table
    else:    
        mslist = ms_table['mz']
    if not 'bonds' in list(ms_table.keys()):
        bslist = [0]*len(mslist)
    else:
        bslist = ms_table['bonds']
    if not 'BDE' in list(ms_table.keys()):
        BDE = [80]*len(mslist)
    else:
        BDE = ms_table['BDE']
    if not 'mid' in list(ms_table.keys()):
        mid = [0]*len(mslist)
    else:
        mid = ms_table['mid']       
            
    got_ms2 = []
    for m,b,bde,_ in zip(mslist,bslist,BDE,mid):
        if bde == 0:
            # bde = np.mean(BDE)
            bde = 80
        delta = abs(refms2['mz'] - m)
        refms2['mserror'] = delta
        if (delta < mserror).any():
            temp = refms2[refms2['mserror'] < mserror].copy()
            temp['BDE'] = bde
            temp['bonds'] = str(b)
            temp['mid'] = _
            got_ms2.append(temp)
            
    if len(got_ms2)>0:
        got_ms2 = pd.concat(got_ms2)
        got_ms2['mz'] = got_ms2['mz'].round(3)
        got_ms2 = got_ms2.sort_values(by='BDE',ascending=True)
        got_ms2 = got_ms2.drop_duplicates(subset=['mz']).reset_index(drop=True)
        got_ms2['intensity'] = got_ms2['intensity']/np.sqrt(got_ms2['BDE'])*10
    return got_ms2


def get_fragments(mol,bonds_comb,adduct,mode):
    H = 1.007825
    Na = 22.98977
    K = 38.963707
    Cl = 34.968853
    if 'Na' in adduct:
        adct = Na
    elif 'K' in adduct:
        adct = K
    elif 'Cl' in adduct:
        adct = -1*Cl
    else:
        adct = H
    fragments = []
    frag_weights = []
    all_bonds = []
    symb_val_rank = {'C':0,'O':2,'N':1,'P':0,'S':0,'na':0}
    for bd in bonds_comb:
        if not type(bd) == list:
            bd = [bd]
        elif not len(bd) > 0:
            continue
        bd = list(set(bd))
        bateat = [(mol.GetBondWithIdx(b).GetBeginAtom(),mol.GetBondWithIdx(b).GetEndAtom()) for b in bd]
        atoms_idx = [[a.GetIdx()for a in be] for be in bateat]
        atoms_symb = [[a.GetSymbol()for a in be] for be in bateat]
        atoms_symb = [[a if a in symb_val_rank.keys() else 'na' for a in be ] for be in bateat]
        atoms_ms = [[a.GetMass()for a in be] for be in bateat]
        atoms_val = [[symb_val_rank[be[0]]-symb_val_rank[be[1]],symb_val_rank[be[1]]-symb_val_rank[be[0]]]for be in atoms_symb]
        # atoms_val = [[be[0]-be[1],be[1]-be[0]]for be in atoms_ms]
        val_dict = {a:b for aa,bb in zip(atoms_idx,atoms_val) for a,b in zip(aa,bb) }
        fmol = Chem.FragmentOnBonds(mol,bd) 
        try:
            frag_mols = Chem.GetMolFrags(fmol,asMols=True) 
        except Exception as e:
            print(e)
            continue
        frag_smarts = [Chem.MolToSmarts(f) for f in frag_mols]
        
        if mode == '-':
            n_charges = [rdMolDescriptors.CalcNumHeteroatoms(mol) for f in frag_mols]
        else:
            n_charges = [1 for f in frag_mols]
        n_Hs = [sum([a.GetNumImplicitHs() for a in f.GetAtoms()]) for f in frag_mols] 
        n_ids = [re.findall("\d+#0",s) for s in frag_smarts]
        n_ids = [[eval(s.replace('#0','')) for s in n] for n in n_ids]
        n_ids = [n if n else [0] for n in n_ids] 

        n_vals = [[-1*val_dict[s] for s in n] for n in n_ids]
        n_breaks = [len(re.findall("-.\d*#0",s))+ 2*len(re.findall("=.\d*#0",s)) for s in frag_smarts]
  
        n_atoms = [i.GetNumAtoms() for i in frag_mols]
        
        fw = []
        ff = []
        ab = []
        for i in range(len(frag_mols)):
            if n_charges[i] == 0: # charged or not
                continue
            
            a = n_breaks[i]
            vals = n_vals[i]
            vals = [int(v/abs(v)) if v!=0 else int(v) for v in vals]
            if n_atoms[i] > 2: 
                if n_Hs[i] > 0: 
                    b = [0] + [-1*(j+1) for j in range(a)] + [(j+1) for j in range(a)]
                else:
                    b = [0] + [(j+1) for j in range(a)] 
                if min(n_atoms)>2: 
                    for v in vals:
                        if v == -1:
                            b.remove(max(b))
                        elif v == 1:
                            b.remove(min(b))
            else:
                # CH3 HO NH2
                b = [0] + [(j+1) for j in range(a)]
            ab.append(b)
        if len(ab) == 2:
            ab_ = itertools.product(ab[0],ab[1])
        elif len(ab) == 3:
            ab_ = itertools.product(ab[0],ab[1],ab[2])
        elif len(ab) == 4:
            ab_ = itertools.product(ab[0],ab[1],ab[2],ab[3])
        else:
            continue
        # ab_ is the possible combination of all HR
    
        for a_b_ in ab_:
            if not sum(a_b_) == 0 :
                continue
            fw_ = [rdMolDescriptors.CalcExactMolWt(frag_mols[i])+a_b_[i]*H for i in range(len(a_b_))]
            ff_ = [frag_mols[i]for i in range(len(a_b_))]
            ff_ = [Chem.MolToSmarts(f) for f in ff_]
            ff_ = [s.replace(p,"") for s in ff_ for p in  re.findall("..\d*#0.",s)]
            if not max(fw_)>=50:
                continue
            fw.extend(fw_)
            ff.extend(ff_)
        fragments.append(ff)
        frag_weights.append(fw)
        all_bonds.append(bd)
    frag_weights.append([rdMolDescriptors.CalcExactMolWt(mol)]) # add the precursor ion
    fragments.append([Chem.MolToSmarts(mol)])
    all_bonds.append([])
    if mode == '-':
        frag_weights = [[f - adct for f in fw] for fw in frag_weights]
    elif mode == '+':
        frag_weights = [[f + adct for f in fw] for fw in frag_weights]
    return fragments,frag_weights,all_bonds




def enu_com(l):
    l = [i+999 for i in l]
    k = len(l)
    choices = [list(np.binary_repr(i,k)) for i in range(1,2**k)]
    combinations = [np.array([eval(i) for i in a])*np.array(l) for a in choices]
    combinations = [[i for i in a if i!=0] for a in combinations]
    combinations = [[i-999 for i in a] for a in combinations]
    combinations = [[int(i) for i in c] for c in combinations]
    return combinations

def get_bridge_bonds(bonds_in_r):
    bb = []
    for i in range(len(bonds_in_r)):
        bs1 = bonds_in_r[i]
        bs2 = [b for j in range(len(bonds_in_r)) if i!=j for b in bonds_in_r[j]]
        for b in bs1:
            if b in bs2:
                bb.append(b)
    bb = set(bb)
    return bb

def bondscomb2(bonds1,bonds2):
    b2 = []
    if bonds1 == bonds2:
        for i in range(len(bonds1)):
            for j in range(i+1,len(bonds1)):
                if not type(bonds1[i]) == list:
                    bonds1[i] = [bonds1[i]]
                if not type(bonds1[j]) == list:
                    bonds1[j] = [bonds1[j]]
                b2.append(bonds1[i]+bonds1[j])
    else:
        for i in bonds1:
            if not type(i) == list:
                i = [i]
            elif len(i) == 0:
                continue
            for j in bonds2:
                if not type(j) == list:
                    j = [j]
                elif len(j) == 0:
                    continue
                if not i == j:
                    b2.append(i+j)
    return b2

def cal_BDE(mol,bond_id):
    e = 80
    if type(bond_id) is int:
        bond = mol.GetBondWithIdx(bond_id)
        tp = bond.GetBondTypeAsDouble()
        b_at = bond.GetBeginAtom()
        e_at = bond.GetEndAtom()
        symb = [b_at.GetSymbol(),e_at.GetSymbol()]
        b_neib = [a.GetIdx() for a in b_at.GetNeighbors()]
        b_neib.remove(e_at.GetIdx())
        e_neib = [a.GetIdx() for a in e_at.GetNeighbors()]
        e_neib.remove(b_at.GetIdx())
        b_bonds_of_neib = [mol.GetBondBetweenAtoms(b,b_at.GetIdx()).GetBondTypeAsDouble() for b in b_neib]
        e_bonds_of_neib = [mol.GetBondBetweenAtoms(b,e_at.GetIdx()).GetBondTypeAsDouble() for b in e_neib]
        if bond.IsInRing():
            if symb == ['O','O']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: # double bonds
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 80 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80
            elif symb == ['C','O'] or symb == ['O','C']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 80
                    elif ave_b > 1 and ave_e > 1:
                        e = 100
                    elif ave_b == 1 and ave_e > 1:
                        e = 80
                    else:
                        e = 60
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80
            elif symb == ['C','N'] or symb == ['N','C']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 80 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80
            elif symb == ['N','O'] or symb == ['O','N']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 80 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80                        
            elif symb == ['C','C']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 105
                    elif ave_b > 1 and ave_e > 1:
                        e = 130
                    elif ave_b == 1 and ave_e > 1:
                        e = 105
                    else:
                        e = 72 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 72 
            elif symb == ['N','N']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 80 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80
                        
        # chain bonds                
        else:
            if symb == ['O','O']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 80 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80
            elif symb == ['C','O'] or symb == ['O','C']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 85 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 85
            elif symb == ['C','N'] or symb == ['N','C']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 80 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80
            elif symb == ['N','O'] or symb == ['O','N']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 80 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 80                        
            elif symb == ['C','C']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 90 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 90 
            elif symb == ['N','N']:
                if min(len(b_bonds_of_neib),len(e_bonds_of_neib)) > 0:
                    ave_b = np.mean(b_bonds_of_neib)
                    ave_e = np.mean(e_bonds_of_neib)
                    if ave_b > 1 and ave_e == 1: 
                        e = 100
                    elif ave_b > 1 and ave_e > 1:
                        e = 120
                    elif ave_b == 1 and ave_e > 1:
                        e = 100
                    else:
                        e = 90 
                else:
                    ave_ = np.mean(b_bonds_of_neib) if len(b_bonds_of_neib)>0 else np.mean(e_bonds_of_neib)
                    if ave_ > 1:
                        e = 100
                    else:
                        e = 90
        e = e * tp
        return e
    elif type(bond_id) is list:
        return [cal_BDE(mol, b) for b in bond_id]
def break_all2(mol,mode='-',adduct=''):
    fragments = []
    frag_weigths = []
    all_bonds = []
    ri = mol.GetRingInfo() # ring information
    bonds_in_r = ri.BondRings()
    bridge_bonds = get_bridge_bonds(bonds_in_r) # dont cleavage brige bonds
    bonds_in_r = [[b_ for b_ in b if b_ not in bridge_bonds] for b in bonds_in_r]
    
    chain_bonds = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing() and b.GetBondTypeAsDouble()<=2]
    ring_bonds = [[[xza[i],xza[j]] for i in range(len(xza)) for j in range(i,len(xza)) if i!=j] for xza in bonds_in_r] 
    
    chain_comb = bondscomb2(chain_bonds,chain_bonds) # comb1:single + singl 
    ring_comb = [bondscomb2(ring_bonds[i],ring_bonds[j]) for i in range(len(ring_bonds)) for j in range(i,len(ring_bonds)) if i!=j]
    ri_ch_comb = [bondscomb2(chain_bonds,i) for i in ring_bonds] # comb2: single + 2* ring
    
    

    ring_bonds = [b for bs in ring_bonds for b in bs] 
    ring_comb = [b for bs in ring_comb for b in bs] 
    ri_ch_comb = [b for bs in ri_ch_comb for b in bs]
    bonds_comb = chain_bonds + ring_bonds + chain_comb + ring_comb + ri_ch_comb
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
    bonds_comb = pop_list(bonds_comb)
    fragments,frag_weigths,all_bonds = get_fragments(mol,bonds_comb,adduct=adduct,mode=mode)
    return fragments,frag_weigths,all_bonds

def multi_break2(mol,mode='-',adduct=''):
    re_ring = {'flavonids':Chem.MolFromSmarts('[#6]1~[#6]~[#6](=[#8])~[#6]~[#6]~[#8]~1'),
        }
    targets = {'O':Chem.MolFromSmarts('[#8R]'),
               'CO':Chem.MolFromSmarts('[#6]=[#8]')
        }
    limit = {'paraC-O':Chem.MolFromSmarts('[#8]=[#6]~[#6]~[#6]-[#8]')}
    mollist = [mol]
    frags = []
    molids = []
    frag_weigths = []
    all_bonds = []
    def re_link(mol):
        Chem.Kekulize(mol,clearAromaticFlags=True)
        m_ = Chem.RWMol(mol)
        atoms = mol.GetSubstructMatch(re_ring['flavonids'])
        atoms_limt = mol.GetSubstructMatches(limit['paraC-O'])
        target_atoms = [mol.GetSubstructMatches(p) for k,p in targets.items()]
        target_atoms = [_ for at in target_atoms for a in at for _ in a]
        atoms = [m_.GetAtomWithIdx(i) for i in set(atoms).intersection(set(target_atoms))]
        atoms = [a for a in atoms if a.IsInRing()]
        atnebs = [[i.GetIdx() for i in a.GetNeighbors() if i.IsInRing()] for a in atoms]
        cent_id = [a.GetIdx() for a in atoms]
        cent_id = [set(cent_id).intersection(set(a)) for a in atoms_limt]
        cent_id = list([c for c in cent_id if len(c)==2][0]) # para- atom
        
        comb1 = [atnebs[0][0],cent_id[0],atnebs[0][1]]
        comb2 = [atnebs[1][0],cent_id[1],atnebs[1][1]]
        
        # The order of the middle two atoms may be reversed and return a null value, so it needs to be removed
        break_bonds = [m_.GetBondBetweenAtoms(comb1[0],comb1[1]),
                       m_.GetBondBetweenAtoms(comb1[2],comb1[1]),
                       m_.GetBondBetweenAtoms(comb2[0],comb2[1]),
                       m_.GetBondBetweenAtoms(comb2[2],comb2[1]),
                       m_.GetBondBetweenAtoms(comb1[0],comb2[1]),
                       m_.GetBondBetweenAtoms(comb1[2],comb2[1]),
                       m_.GetBondBetweenAtoms(comb2[0],comb1[1]),
                       m_.GetBondBetweenAtoms(comb2[2],comb1[1]),
                       ]
        break_bonds = [b for b in break_bonds if b]
        break_bonds = [b.GetIdx() for b in break_bonds]
        connet_bonds = [(comb1[0],comb1[2]),
                        (comb2[0],comb2[2])
                        ]
        m_.AddBond(connet_bonds[0][0],connet_bonds[0][1],Chem.BondType.SINGLE)
        m_.AddBond(connet_bonds[1][0],connet_bonds[1][1],Chem.BondType.SINGLE)
        fmol = Chem.FragmentOnBonds(m_,break_bonds)
        m_ = Chem.RWMol(fmol) 
        # frag_mols = Chem.GetMolFrags(fmol,asMols=True)
        # fw = [rdMolDescriptors.CalcExactMolWt(m) for m in frag_mols]
        # m_ = frag_mols[fw.index(max(fw))]
        # m_ = Chem.RWMol(m_)
        free = m_.GetSubstructMatches(AllChem.MolFromSmarts('[#0]'))
        free = [_ for ff in free for _ in ff]
        free_neib = [m_.GetAtomWithIdx(i).GetNeighbors() for i in free]
        free_neib = [_ for ff in free_neib for _ in ff]
        free_neib = [f.GetIdx() for f in free_neib]
        for i in range(len(free)):
            assert len(free) == len(free_neib)
            m_.RemoveBond(free[i],free_neib[i])
            
        frag_mols = Chem.GetMolFrags(m_,asMols=True)
        fw = [rdMolDescriptors.CalcExactMolWt(m) for m in frag_mols]
        m_ = frag_mols[fw.index(max(fw))]
        Chem.Kekulize(m_,clearAromaticFlags=True)
        return m_
    if mol.HasSubstructMatch(re_ring['flavonids']):
        try:
            m_ = re_link(mol)
            if m_:
                mollist.append(m_)
        except Exception as e:
            print(e,'Can‘t Get Substructure!!!')
            pass
    for i,mol in enumerate(mollist):
        a,b,c = break_ht2(mol,mode,adduct)
        frags.extend(a)
        frag_weigths.extend(b)
        all_bonds.extend(c)
        molids.extend([i]*len(b))
    mollist = [Chem.MolToSmiles(m) for m in mollist]
    # mol_smiles = [mollist[i] for i in molids]
    return frags,molids,frag_weigths,all_bonds    

def break_ht2(mol,mode='-',adduct='',subclass='',CH3=False):
    patterns = {
    'N': AllChem.MolFromSmarts('[#7]'),          
    'O': AllChem.MolFromSmarts('[OD2;!R]'), # -O-
    'P': AllChem.MolFromSmarts('[#15;!R]'),
    'S': AllChem.MolFromSmarts('[#16;!R]'),
    'CO': AllChem.MolFromSmarts('[CX3]=O'), # CO
    }
    patterns1 = {
    'R-R': AllChem.MolFromSmarts('[#6]1(~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]~2)~[#6]~[#6]~[#6]~[#6]~[#6]~1'), # ph-ph
    'RCR': AllChem.MolFromSmarts('[#6]1(~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]~2)~[#6]~[#6]~[#6]~[#6]~[#6]~1'), # ph-C-ph
    'RCCR': AllChem.MolFromSmarts('[#6]1(~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]~2)~[#6]~[#6]~[#6]~[#6]~[#6]~1'), # ph-C-C-ph
    }
    if CH3:
        patterns2 = {
            'OH': AllChem.MolFromSmarts('[OX2H1]'),
            'CH3': AllChem.MolFromSmarts('[CH3]'),
            }
    else:
        patterns2 = {
            'OH': AllChem.MolFromSmarts('[OX2H1]'),
            }
    patterns3 = {'C': AllChem.MolFromSmarts('[#6D3R]-[#6R]')} 
    patterns4 = {'char2':AllChem.MolFromSmarts('[#8]=[#6](-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1)/[#6]=[#6]/[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'),
        }
    ring_zares = {'cch':AllChem.MolFromSmarts('[#6;!R]=[#6;!R]-[#8H;!R]')
        }
    ring_za = {'Or6': AllChem.MolFromSmarts('[#6]1~[#6]~[#6]~[#6]~[#6]~[#8]~1'), 
               'Or5': AllChem.MolFromSmarts('[#6]1~[#6]~[#6]~[#6]~[#8]~1'), 
               'CO': AllChem.MolFromSmarts('[#6D3R]=[#8]'), 
               }

    ring_res = {'xch':AllChem.MolFromSmarts('[#6]1=[#6](-[#8H])-[#6]=[#6]-[#6]=[#6]-1'), 
                'xcm':AllChem.MolFromSmarts('[#8D2;!R]-[#6]1-[#6]=[#6](-[#8D2;!R])-[#6]=[#6]-[#6]=1'), 
                'RDA':AllChem.MolFromSmarts('[#6]1-[#6]-[#6]-[#6]=[#6](-[#8H])[#8]-1'),
                'RDA2':AllChem.MolFromSmarts('[#6]1-[#6]-[#6]-[#6](-[#8H])=[#6]-[#8]-1'),
                }
    fragments = []
    frag_weigths = []
    all_bonds = []
    # mol =  Chem.RWMol(mol)
    ri = mol.GetRingInfo() # ring information
    bonds_in_r = ri.BondRings()
    bridge_bonds = get_bridge_bonds(bonds_in_r) 
    
    
    x1 = [mol.GetSubstructMatches(p) for k,p in patterns.items()] 
    x1_ = [mol.GetSubstructMatches(p) for k,p in patterns1.items()] 
    x2 = [mol.GetSubstructMatches(p) for k,p in patterns2.items()] 
    x2 = [a[:2] for a in x2]
    x3 = [mol.GetSubstructMatches(p) for k,p in patterns3.items()] 
    x4 = [mol.GetSubstructMatches(p) for k,p in patterns4.items()]
    xrzares = [mol.GetSubstructMatches(p) for k,p in ring_zares.items()]
    xrza = [mol.GetSubstructMatches(p) for k,p in ring_za.items()] 
    xresr = [mol.GetSubstructMatches(p) for k,p in ring_res.items()] 
    x = [a for b in x1+x1_+x2 for a in b]
    x4 = [a for b in x4 for a in b]
    xb = [a for b in x3 for a in b]
    xbb = [a for b in x1_ for a in b] 
    xrzares_ = [a for b in xrzares for a in b]
    xrza_ = [a for b in xrza for a in b]
    xresr_ = [a for b in xresr for a in b]
    x = [a for at in x  for a in at if at]
    x = set(x)
    xrzares_ = [a for at in xrzares_  for a in at if at]
    xrzares_ = set(xrzares_) 
    
    atoms1 = [mol.GetAtomWithIdx(i) for i in x]
    bonds1 = [at.GetBonds() for at in atoms1]
    bonds1 = [b for bs in bonds1 for b in bs]
    bonds1 = [b for b in bonds1 if not b.IsInRing()] 
    bonds1 = [b for b in bonds1 if b.GetBondTypeAsDouble()==1] 
    bonds3 = [mol.GetBondBetweenAtoms(b[0],b[1]) for b in xb]
    bonds3 = [b for b in bonds3 if not b.IsInRing()] #C-C
    atoms3 = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in bonds3]
    atoms3 = [a for at in atoms3 for a in at] 
    
    # ring bonds
    xrza__ = []
    for xr in xrza_:
        if len(xr) > 4:
            if set(xr).intersection(set(atoms3)):
                xrza__.append(xr)
        else:
            xrza__.append(xr)
            
    bonds4 = [[mol.GetBondBetweenAtoms(at,bt) for at in a for bt in a] for a in xrza__]
    bonds4 = [[b for b in bs if b] for bs in bonds4]
    bonds4 = [[b for b in bs if b.GetBeginAtom().GetNumImplicitHs()+b.GetEndAtom().GetNumImplicitHs()<=3] for bs in bonds4]
    bonds4 = [[b for b  in bs if b.IsInRing()] for bs in bonds4]
    bonds4 = [[b for b  in bs if b.IsInRing() and b.GetBondTypeAsDouble()==1] for bs in bonds4]
    
    atoms5 = [[mol.GetAtomWithIdx(i) for i in ats] for ats in xresr_]
    # atoms5 = [[a for a in ats if a.GetSymbol()=='C' and True in [x.GetSymbol()=='O' for x in a.GetNeighbors()]] for ats in atoms5]
    bonds5_0 = [[mol.GetBondBetweenAtoms(at,bt) for at in a for bt in a] for a in xresr_]
    bonds5_0 = [[b for b in bs if b] for bs in bonds5_0]
    # bonds5_0 = [[b for b in bs if b.GetBondTypeAsDouble()==1] for bs in bonds5_0]
    bonds5 = [[a.GetBonds() for a in at] for at in atoms5]
    bonds5 = [[b_  for b in bs for b_ in b]for bs in bonds5]
    bonds5 = [b0+b1 for b0,b1 in zip(bonds5_0,bonds5)] 
    bonds5 = [[b for b in bs if b.IsInRing()] for bs in bonds5]#
    atoms6 = [mol.GetAtomWithIdx(i) for i in x4] #
    bonds6 = [at.GetBonds() for at in atoms6]
    bonds6 = [b for bs in bonds6 for b in bs]
    bonds6 = [b for b in bonds6 if not b.IsInRing()] #
    bonds6 = [b for b in bonds6 if b.GetBondTypeAsDouble()==2] #
    # 整合b4 b5
    bonds4_ = []
    for b in bonds4+bonds5: 
        bonds4_.append([x for x in b if x])

    idx = [b.GetIdx() for b in bonds1+bonds6] 
    idx = list(set(idx))

    idx_rza = [[b.GetIdx() for b in bs] for bs in bonds4_] 
    idx_rza = [[b for b in bs if not b in bridge_bonds] for bs in idx_rza] 
    
    idx2_2 = []
    idx_rza_comb = []
    idx = list(set(idx))
    idx2_1 = bondscomb2(idx,idx) 
    if len(idx_rza)>0:
        idx_rza = [list(set(i)) for i in idx_rza if len(i)>0]
        idx_rza = pop_list(idx_rza)
        idx_rza = [bondscomb2(i,i) for i in idx_rza]
        idx_rza_comb = [bondscomb2(idx_rza[i],idx_rza[j]) for i in range(len(idx_rza)) for j in range(i,len(idx_rza)) if i!=j] 
        idx2_2 = [bondscomb2(idx,i) for i in idx_rza ] 
        temp = []
        for b2s in idx_rza:
            temp += b2s
        idx_rza = temp
        temp = []
        for b2s in idx_rza_comb:
            temp += b2s
        idx_rza_comb = temp
        temp = []
        for b2s in idx2_2:
            temp += b2s
        idx2_2 = temp
            
    idx2 = idx2_1+idx2_2 
    bonds_comb = idx + idx_rza + idx_rza_comb + idx2
    bonds_comb = pop_list(bonds_comb)
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
    fragments,frag_weigths,all_bonds = get_fragments(mol,bonds_comb,adduct=adduct,mode=mode)
    return fragments,frag_weigths,all_bonds


def fragments_generation(smiles,mode = '+',t=None):
    adduct = 'H'
    sick_mols = []
    mol = AllChem.MolFromSmiles(smiles)
    try:
        Chem.Kekulize(mol,clearAromaticFlags=True)
    except Exception as e:
        print(e)
        sick_mols.append(mol)
        temp_ = pd.DataFrame({'mz':[0],'smarts':[0],'bonds':[0],'mid':[0]})
        return temp_
    if t == 't3':
        try:
            f,fw,bs = break_t32(mol,mode=mode,adduct=adduct) 
            # mol_smiles = [sms]*len(fw)
            mids = [0]*len(fw)
        except Exception as e:
            sick_mols.append(mol)
            temp_ = pd.DataFrame({'mz':[0],'smarts':[0],'bonds':[0],'mid':[0]})
            return temp_
    elif t == 'ht':
        try:
            f,mids,fw,bs = multi_break2(mol,adduct=adduct,mode=mode)
            mids = [0]*len(fw)
            # mol_smiles = [sms]*len(fw)
        except Exception as e:
            sick_mols.append(mol)
            temp_ = pd.DataFrame({'mz':[0],'smarts':[0],'bonds':[0],'mid':[0]})
            return temp_
    else:
        try:
            f,fw,bs = break_all2(mol,adduct=adduct,mode=mode)
            mids = [0]*len(fw)
            # mol_smiles = [sms]*len(fw)
        except Exception as e:
            sick_mols.append(mol)
            temp_ = pd.DataFrame({'mz':[0],'smarts':[0],'bonds':[0],'mid':[0]})
            return temp_
      
    mz = []
    fragsmts = []
    bonds = []
    molids = []
    for f_,m_,b_,i_ in zip(f,fw,bs,mids):
        for _1,_2 in zip(f_,m_):
            fragsmts.append(_1)
            mz.append(np.round(_2,3))
            bonds.append(b_)
            molids.append(i_)

    temp_ = pd.DataFrame([mz,fragsmts,bonds,molids]).T 
    temp_.columns = ['mz','smarts','bonds','mid'] 
    temp_ = temp_.drop_duplicates(subset=['mz']).dropna()
    temp_ = temp_.reset_index(drop=True)
    return temp_

def insiloscore2(ms_table,refms2,mserror=20e-3):
    refms2 = refms2.copy()
    refms2['intensity'] = refms2['intensity']/refms2['intensity'].sum()
    if len(ms_table.shape) == 1:
        mslist = ms_table
    else:    
        mslist = ms_table['mz']
    if not 'bonds' in list(ms_table.keys()):
        bslist = [0]*len(mslist)
    else:
        bslist = ms_table['bonds']
    if not 'BDE' in list(ms_table.keys()):
        BDE = [80]*len(mslist)
    else:
        BDE = ms_table['BDE']
    if not 'mid' in list(ms_table.keys()):
        mid = [0]*len(mslist)
    else:
        mid = ms_table['mid']       
            
    got_ms2 = []
    for m,b,bde,_ in zip(mslist,bslist,BDE,mid):
        if bde == 0:
            # bde = np.mean(BDE)
            bde = 80
        delta = abs(refms2['mz'] - m)
        refms2['mserror'] = delta
        if (delta < mserror).any():
            temp = refms2[refms2['mserror'] < mserror].copy()
            temp['BDE'] = bde
            temp['bonds'] = str(b)
            temp['mid'] = _
            got_ms2.append(temp)
            
    if len(got_ms2)>0:
        got_ms2 = pd.concat(got_ms2)
        got_ms2['mz'] = got_ms2['mz'].round(3)
        got_ms2 = got_ms2.sort_values(by='BDE',ascending=True)
        got_ms2 = got_ms2.drop_duplicates(subset=['mz']).reset_index(drop=True)
        #got_ms2['intensity'] = got_ms2['intensity']/np.sqrt(got_ms2['BDE'])*10
        count = got_ms2['intensity'].sum()
        mes = sum(np.e**(-0.5*(got_ms2['mserror']/mserror)**2))/len(refms2)
    else:
        count = 0
        mes = 0
    if len(mslist) > 0:
        punish = (len(mslist)-len(got_ms2))/len(mslist)
    else:
        punish = 1
    # count = count*(1-punish)
    n = len(got_ms2)
    return punish,count,mes,got_ms2



columns = ['NAME','PRECURSORMZ','PRECURSORTYPE','FORMULA',
       'Ontology','INCHIKEY','SMILES','RETENTIONTIME','CCS',
       'IONMODE','INSTRUMENTTYPE','INSTRUMENT','COLLISIONENERGY',
       'Comment','Num Peaks']
def parse_MSDIAL(df,iso=False,maxlen=10,tolerance=5e-3,keepms2=False):
    """
    parse MS-DIAL exported files for masskg
    """
    df_ = df.copy()
    if "Ms1" in df:
        MS1 = df_['Ms1']
        MS2 = df_['Ms2']
        if keepms2:
            feature_table = df_[['ID','Rt','Mz','Size','Ms2']]
        else:
            feature_table = df_[['ID','Rt','Mz','Size']]
    elif "Precursor m/z" in df_:
        MS1 = df_['MS1 isotopes']
        MS2 = df_['MSMS spectrum']
        if keepms2:
            feature_table = df_[['PeakID','RT (min)','Precursor m/z','Area','MSMS spectrum']]
        else:
            feature_table = df_[['PeakID','RT (min)','Precursor m/z','Area']]
    elif "Precursor mz" in df_:
        MS1 = df_['MS1 isotopes']
        MS2 = df_['MSMS spectrum']
        if keepms2:
            feature_table = df_[['PeakID','RT (min)','Precursor mz','Area','MSMS spectrum']]
        else:
            feature_table = df_[['PeakID','RT (min)','Precursor mz','Area']]
    else:
        raise KeyError("Keys for MS1 were not found!")
    if keepms2:
        feature_table.columns = ['ID','Rt','Mz','Size','MS2']
    else:
        feature_table.columns = ['ID','Rt','Mz','Size']
    if 'Adduct' in df_:
        feature_table = pd.concat([feature_table,df_['Adduct']],axis=1)

    extra_ms2 = feature_table[feature_table['Mz'].isnull()]['ID']
    feature_table = feature_table[~feature_table['Mz'].isnull()]
    
    if len(extra_ms2) > 0:

        for i,m in extra_ms2.items():

            if type(MS2[i-1]) == str:
                MS2[i-1] += str(m)
                last_idx = i-1
            else:
     
                MS2[last_idx] += m
            
                
            

    MS2 = MS2[feature_table.index]

    ms2_list = get_ms(MS2,maxlen=maxlen)
    iso_list = []
    
    if iso:
        ms1_list = get_ms(MS1)

        for i in range(len(df)):
            m = feature_table['Mz'][i]
            pk = ms1_list[i]
            delta_mz = pk['mz']-m
            wanted0 = abs(delta_mz) < tolerance
            wanted1 = delta_mz.between(0.997-tolerance,1.006+tolerance)
            wanted2 = delta_mz.between(1.994-tolerance,2.013+tolerance)
            w0 = sum(pk[wanted0]['intensity'])
            w1 = sum(pk[wanted1]['intensity'])
            w2 = sum(pk[wanted2]['intensity'])
            if not w0 == 0:
                w1 = w1/w0
                w2 = w2/w0
                w0 = w0/w0
            m0 = m
            m1 = pk[wanted1]['mz'].mean()
            m2 = pk[wanted2]['mz'].mean()
            iso_list.append(pd.DataFrame([[m0,w0],[m1,w1],[m2,w2]],columns=['mz','intensity']))
    return iso_list,ms2_list,feature_table


def msptranslator(path):
    """
    parsing a msp file by dereplication, where the ms2 peaks of
    compounds sharing a same name would be merged.

    """
    MSlist2 = []
    res = []
    feature_table = dict()
    ms2 = []
    with open(path,encoding='utf-8') as f:
        raw_data=f.readlines()# read mgf file to a list line by line
        
    for s in raw_data:
        if re.match('[A-Za-z]',s): 
            s = s.replace('\n', '')
            s = s.split(":",1)
            feature_table[s[0]] = s[1]
        elif re.match('\d+',s):
            s = s.replace('\t',':').replace('\n', '')
            s = s.split(":",1)
            if len(s) == 2:
                ms2.append(s)
            else:
                print(s)
        else:
            res.append(feature_table)
            MSlist2.append(np.array(ms2,dtype=float))
            feature_table = dict()
            ms2 = []
    res = pd.DataFrame(res).reset_index(drop=True)
    print('total ============= {} blocks finished'.format(len(res)))
    # res = res[columns]
    return res,MSlist2

def prems2(_ms2):
    # profile to centriod
    binsize = 0.5
    _ms2 = pd.DataFrame(_ms2)
    _ms2.columns = ['mz','intensity']
    _ms2 = _ms2.sort_values(by=['intensity'],ascending=False).reset_index(drop=True)
    res = []
    for i in range(_ms2.shape[0]):
        mz = list(_ms2['mz'])[i]
        temp = _ms2[(_ms2['mz']>mz-binsize)&(_ms2['mz']<mz+binsize)].iloc[0]
        res.append(temp)
    res = pd.DataFrame(res)
    res = res.drop_duplicates().sort_values(by=['mz']).reset_index(drop=True)
    res['intensity'] = res['intensity']/res['intensity'].max()*10000
    res['intensity'] = res['intensity'].round(1)
    res = res.reset_index()
    res.columns = ['ID','mz','intensity']
    return res

def get_rank(query,test_ms,mode,fragpath,dbparams):
    pepmass = query.iloc[0]['pepmass']
    client = myclient('masskg')
    candidate_db = dbparams[0]
    mstable = dbparams[1]
    ms2_list = dbparams[2]
    if mode == '+':
         fragpath = os.path.join(fragpath,'POS')
    elif mode == '-':
         fragpath = os.path.join(fragpath,'NEG')
   
    test_ms['intensity'] = test_ms['intensity']/test_ms['intensity'].max()
    res = []
    got_ms = []
    got_fms = []
    got_mols = []
    for j in range(query.shape[0]):
        temp_test_ms = test_ms.copy()
        fm = query.iloc[j]['Formula']
        adduct = query.iloc[j]['Adduct']

        # level.1 experimental database
        css = mstable[(mstable['FORMULA']==fm)&(mstable['PRECURSORTYPE']==adduct)]
        css_output = css.copy()[['NAME','INCHIKEY','SMILES',
                                 'Kingdom', 'Superclass', 'Class', 'Subclass']]
        css_output.columns = ['Chemical Name','InChIKey','SMILES',
                                 'Kingdom', 'Superclass', 'Class', 'Subclass']
        
        candims = [ms2_list[i] for i in css['ID'].astype(int)]
        css_output = css_output[[len(c)>0 for c in candims]]
        candims = [c for c in candims if len(c)>0]
        if len(candims) > 0:
            in_siloscores = [insiloscore(h,temp_test_ms,mserror=20e-3) for h in candims]
            n_matched = [s[0] for s in in_siloscores]
            struct_score = [s[1] for s in in_siloscores]
            frag_score = [s[2] for s in in_siloscores]
    
            formu_info =  pd.concat([query.iloc[j]]*len(css_output),axis=1).T
            formu_info = formu_info.reset_index(drop=True)
            css_output = css_output.reset_index(drop=True)
            css_output = pd.concat([formu_info,css_output],axis=1)
            css_output['punish_score'] = 1
            css_output['matched_peaks'] = n_matched
            css_output['structure_score'] = struct_score
            css_output['fragmes_mse'] = frag_score
            css_output['level'] = 1

            css_output = css_output[(css_output['matched_peaks']>=1)&css_output['structure_score']>=0.3]
        if len(css_output) > 0:
            res.append(css_output) 
            got_ms.extend([in_siloscores[i][3] for i in css_output.index])
            got_fms.extend([fm]*len(in_siloscores))
            got_mols.extend([0]*len(css_output))
        else:    

            if adduct == 'Na':
                temp_test_ms['mz'] = test_ms['mz'] + 1.007825 - 22.989770
            elif adduct == 'Cl':
                temp_test_ms['mz'] = test_ms['mz'] - 1.007825 - 34.968853
            elif adduct == 'K':
                temp_test_ms['mz'] = test_ms['mz'] + 1.007825 - 38.963708
            # level.2 insilico database
            # cs = candidate_db[candidate_db['FORMULA']==fm]
            cs = client.searchcds(candidate_db,fm)
            cs_output = cs.copy()[['Exact_mass','NAME','InChIKey','SMILES','MassKGID',
                                   'Kingdom', 'Superclass', 'Class', 'Subclass',]]
            ids = list(cs.MassKGID.astype(str))
            if not len(cs) > 0:
                continue
    
            smiles = cs['SMILES']
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            candims = [pd.read_csv(os.path.join(fragpath,'fragment_database{}\\{}.csv'.format(eval(i)//20000,i))) if not "GENC" in i 
                       else pd.read_csv(os.path.join(fragpath,'genfragment_database\\{}.csv'.format(re.findall('\d+',i)[0]))) for i in ids]
            cs_output = cs_output[[len(c)>0 for c in candims]]
            candims = [c for c in candims if len(c)>0]
            if not len(candims) > 0:
                continue
            mids = [set(m['mid']) for m in candims]
            mollist = [[a] if len(b)==1 else multi_break2(a,onlymol=True) for a,b in zip(mols,mids)]
            in_siloscores = [insiloscore2(h,temp_test_ms,mserror=20e-3) for h in candims]
            punish_score = [s[0] for s in in_siloscores]
            struct_score = [s[1] for s in in_siloscores]
            frag_score = [s[2] for s in in_siloscores]
    
            formu_info =  pd.concat([query.iloc[j]]*len(cs_output),axis=1).T
            formu_info = formu_info.reset_index(drop=True)
            cs_output = cs_output.reset_index(drop=True)
            cs_output = pd.concat([formu_info,cs_output],axis=1)
            cs_output['punish_score'] = punish_score
            css_output['matched_peaks'] = [len(s[3]) for s in in_siloscores]
            cs_output['structure_score'] = struct_score
            cs_output['fragmes_mse'] = frag_score
            cs_output['level'] = 2
            res.append(cs_output)
            got_ms.extend([s[3] for s in in_siloscores])
            got_fms.extend([fm]*len(in_siloscores))
            got_mols.extend(mollist)
            
    if not len(res)>0:
        return []
    
    res = pd.concat(res).reset_index(drop=True)
    res = res[res['structure_score']>0] 
    got_ms = [got_ms[i] for i in res.index]
    got_mols = [got_mols[i] for i in res.index]
    got_fms = [got_fms[i] for i in res.index]

    ngot_ms = []
    for ii in range(len(got_ms)):
        fm = got_fms[ii]
        gm = got_ms[ii]
        gm['formula'] = ''
        if not min(abs(gm['mz']-pepmass)) <= 0.01:
            gm = gm.append(pd.DataFrame({'ID':[999],'mz':[pepmass],'intensity':[0],'mserror':[0],
                                         'BDE':[0],'bonds':[0],'mid':[0],'formula':[fm]}))
        else:
            gm[abs(gm['mz']-pepmass) <= 0.01]['formula'] = fm
        ngot_ms.append(gm)
        

    
    res = res.reset_index(drop=True)
    nloss = [assign_nl2(x, mode) for x in ngot_ms]
    nlscores = []
    for s in nloss:
        if len(s) >0:
            if len(s[1]) > 0:
                ids = pd.concat([s[1]['Source_ID'],s[1]['Target_ID']])
                _s = pd.concat([s[0][s[0]['ID']==i][['intensity','mserror']] for i in ids])
                _s = (np.e**(-0.5*_s['mserror']/0.01)*_s['intensity']).sum()/2
                nlscores.append(_s)
            else:
                nlscores.append(0)
        else:
            nlscores.append(0)
            
    res['nlscore'] = nlscores
    res['final_score'] = 1*res['structure_score'] + 0.5*res['nlscore'] + 0.1*res['formulascore']
    res = res.sort_values(by='final_score',ascending=False)
    # got_ms = [got_ms[i] for i in res.index]
    got_ms = [nloss[i] for i in res.index]
    got_mols = [got_mols[i] for i in res.index]
    res = res.reset_index(drop=True)
    return res,got_ms,got_mols


    
    
def get_rank(query,test_ms,mode,fragpath,dbparams):
    pepmass = query.iloc[0]['pepmass']
    client = myclient('masskg')
    candidate_db = dbparams[0]
    mstable = dbparams[1]
    ms2_list = dbparams[2]
    if mode == '+':
         fragpath = os.path.join(fragpath,'POS')
    elif mode == '-':
         fragpath = os.path.join(fragpath,'NEG')
   
    test_ms['intensity'] = test_ms['intensity']/test_ms['intensity'].max()
    res = []
    got_ms = []
    got_fms = []
    got_mols = []
    for j in range(query.shape[0]):
        temp_test_ms = test_ms.copy()
        fm = query.iloc[j]['Formula']
        adduct = query.iloc[j]['Adduct']

        # level.1 experimental database
        css = mstable[(mstable['FORMULA']==fm)&(mstable['PRECURSORTYPE']==adduct)]
        css_output = css.copy()[['NAME','INCHIKEY','SMILES',
                                 'Kingdom', 'Superclass', 'Class', 'Subclass']]
        css_output.columns = ['Chemical Name','InChIKey','SMILES',
                                 'Kingdom', 'Superclass', 'Class', 'Subclass']
        
        candims = [ms2_list[i] for i in css['ID'].astype(int)]
        css_output = css_output[[len(c)>0 for c in candims]]
        candims = [c for c in candims if len(c)>0]
        if len(candims) > 0:
            in_siloscores = [insiloscore(h,temp_test_ms,mserror=20e-3) for h in candims]
            n_matched = [s[0] for s in in_siloscores]
            struct_score = [s[1] for s in in_siloscores]
            frag_score = [s[2] for s in in_siloscores]
    
            formu_info =  pd.concat([query.iloc[j]]*len(css_output),axis=1).T
            formu_info = formu_info.reset_index(drop=True)
            css_output = css_output.reset_index(drop=True)
            css_output = pd.concat([formu_info,css_output],axis=1)
            css_output['punish_score'] = 1
            css_output['matched_peaks'] = n_matched
            css_output['structure_score'] = struct_score
            css_output['fragmes_mse'] = frag_score
            css_output['level'] = 1


            css_output = css_output[(css_output['matched_peaks']>=1)&css_output['structure_score']>=0.3]
        if len(css_output) > 0:
            res.append(css_output) 
            got_ms.extend([in_siloscores[i][3] for i in css_output.index])
            got_fms.extend([fm]*len(in_siloscores))
            got_mols.extend([0]*len(css_output))
        else:    
            # defualt ion type of generated ion fragments is ±H, here transfer other ion types as  ±H
            if adduct == 'Na':
                temp_test_ms['mz'] = test_ms['mz'] + 1.007825 - 22.989770
            elif adduct == 'Cl':
                temp_test_ms['mz'] = test_ms['mz'] - 1.007825 - 34.968853
            elif adduct == 'K':
                temp_test_ms['mz'] = test_ms['mz'] + 1.007825 - 38.963708
            # level.2 insilico database
            # cs = candidate_db[candidate_db['FORMULA']==fm]
            cs = client.searchcds(candidate_db,fm)
            cs_output = cs.copy()[['Exact_mass','NAME','InChIKey','SMILES','MassKGID',
                                   'Kingdom', 'Superclass', 'Class', 'Subclass',]]
            ids = list(cs.MassKGID.astype(str))
            if not len(cs) > 0:
                continue
    
            smiles = cs['SMILES']
            mols = [Chem.MolFromSmiles(s) for s in smiles]

            candims = []
            for i in ids:
                if "GENCR" in i:
                    candims.append(pd.read_csv(os.path.join(fragpath,'genfragment_database',"R{}.csv".format(re.findall('\d+',i)[0])))) # NPs generated based on NPPS
                elif "GENC" in i:
                    candims.append(pd.read_csv(os.path.join(fragpath,'genfragment_database',"{}.csv".format(re.findall('\d+',i)[0])))) # NPs generated based on coconut
                else:
                    candims.append(pd.read_csv(os.path.join(fragpath,f'fragment_database{eval(i)//20000}',f'{i}.csv')))
            
            
            cs_output = cs_output[[len(c)>0 for c in candims]]
            candims = [c for c in candims if len(c)>0]
            if not len(candims) > 0:
                continue
            mids = [set(m['mid']) for m in candims]
            mollist = [[a] if len(b)==1 else multi_break2(a,onlymol=True) for a,b in zip(mols,mids)]
            in_siloscores = [insiloscore2(h,temp_test_ms,mserror=20e-3) for h in candims]
            punish_score = [s[0] for s in in_siloscores]
            struct_score = [s[1] for s in in_siloscores]
            frag_score = [s[2] for s in in_siloscores]
    
            formu_info =  pd.concat([query.iloc[j]]*len(cs_output),axis=1).T
            formu_info = formu_info.reset_index(drop=True)
            cs_output = cs_output.reset_index(drop=True)
            cs_output = pd.concat([formu_info,cs_output],axis=1)
            cs_output['punish_score'] = punish_score
            css_output['matched_peaks'] = [len(s[3]) for s in in_siloscores]
            cs_output['structure_score'] = struct_score
            cs_output['fragmes_mse'] = frag_score
            cs_output['level'] = 2
            res.append(cs_output)
            got_ms.extend([s[3] for s in in_siloscores])
            got_fms.extend([fm]*len(in_siloscores))
            got_mols.extend(mollist)
            
    if not len(res)>0:
        return []
    
    res = pd.concat(res).reset_index(drop=True)
    res = res[res['structure_score']>0] # filtter candidates structure score = 0, otherwise nl_score cant calc
    got_ms = [got_ms[i] for i in res.index]
    got_mols = [got_mols[i] for i in res.index]
    got_fms = [got_fms[i] for i in res.index]

    ngot_ms = []
    for ii in range(len(got_ms)):
        fm = got_fms[ii]
        gm = got_ms[ii]
        gm['formula'] = ''
        if not min(abs(gm['mz']-pepmass)) <= 0.01:
            gm = gm.append(pd.DataFrame({'ID':[999],'mz':[pepmass],'intensity':[0],'mserror':[0],
                                         'BDE':[0],'bonds':[0],'mid':[0],'formula':[fm]}))
        else:
            gm[abs(gm['mz']-pepmass) <= 0.01]['formula'] = fm
        ngot_ms.append(gm)
        

    
    res = res.reset_index(drop=True)
    nloss = [assign_nl2(x, mode) for x in ngot_ms]
    nlscores = []
    for s in nloss:
        if len(s) >0:
            if len(s[1]) > 0:
                ids = pd.concat([s[1]['Source_ID'],s[1]['Target_ID']])
                _s = pd.concat([s[0][s[0]['ID']==i][['intensity','mserror']] for i in ids])
                _s = (np.e**(-0.5*_s['mserror']/0.01)*_s['intensity']).sum()/2
                nlscores.append(_s)
            else:
                nlscores.append(0)
        else:
            nlscores.append(0)
            
    res['nlscore'] = nlscores
   
    res['final_score'] = 1*res['structure_score'] + 0.5*res['nlscore']+res['fragmes_mse']*1 + 1*res['formulascore']
    res = res.sort_values(by='final_score',ascending=False)
    # got_ms = [got_ms[i] for i in res.index]
    got_ms = [nloss[i] for i in res.index]
    got_mols = [got_mols[i] for i in res.index]
    res = res.reset_index(drop=True)
    return res,got_ms,got_mols


def prems2(_ms2):
    # profile to centriod
    binsize = 0.5
    _ms2 = pd.DataFrame(_ms2)
    _ms2.columns = ['mz','intensity']
    _ms2 = _ms2.sort_values(by=['intensity'],ascending=False).reset_index(drop=True)
    res = []
    for i in range(_ms2.shape[0]):
        mz = list(_ms2['mz'])[i]
        temp = _ms2[(_ms2['mz']>mz-binsize)&(_ms2['mz']<mz+binsize)].iloc[0]
        res.append(temp)
    res = pd.DataFrame(res)
    res = res.drop_duplicates().sort_values(by=['mz']).reset_index(drop=True)
    res['intensity'] = res['intensity']/res['intensity'].max()*10000
    res['intensity'] = res['intensity'].round(1)
    res = res.reset_index()
    res.columns = ['ID','mz','intensity']
    return res