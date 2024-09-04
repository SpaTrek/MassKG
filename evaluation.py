from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
import pandas as pd
import numpy as np
from matchms import calculate_scores
from matchms import Spectrum
from matchms.similarity import CosineGreedy


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


def topk(k,scores):
    temp = [s < k for s in scores]
    bad_ids = [i for i in range(len(scores)) if scores[i]>=k]
    if len(temp)-sum([s>6666 for s in scores]) > 0:
        print('global score',sum(temp)/len(temp),
              '\n with formula score',sum(temp)/(len(temp)-sum([s>6666 for s in scores])))
        return bad_ids,sum(temp)/(len(temp)-sum([s>6666 for s in scores]))
    else:
        print('global score',0,
              '\n with formula score',0)
        return bad_ids,0
    

def ms2sim(truems2,predms2):
    """
    input: trums2 and predms2 are both pd.DataFrame object with columns ['mz','intensity']
    output: a df with cosine simlarity and matched peaks between a list of query ms2 and refms2
    """
    truems2 = truems2.sort_values(by='mz')
    refspectrum = Spectrum(
        mz = truems2['mz'].values,
        intensities=truems2['intensity'].values,
        metadata={"id":0,
                  "precursor_mz":0}
    )

    predms2 = predms2.sort_values(by='mz')
    spectrum = Spectrum(mz=predms2['mz'].values,
                intensities=predms2['intensity'].values,
                metadata={"id": f'spectrum_{i}',
                        "precursor_mz":0})
    similarity_measure = CosineGreedy()
    scores = calculate_scores([refspectrum], [spectrum], similarity_measure) 
    
    simres = []
    for (reference, query, score) in scores:
        simres.append([reference.get('id'),query.get('id'),score[0],score[1]])
    simres = pd.DataFrame(simres)
    simres.columns = ['refid','queryid','simscore','n_matched']
    return simres    

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