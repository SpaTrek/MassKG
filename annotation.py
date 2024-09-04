from rdkit import Chem
import numpy as np
import pandas as pd
import re
import os

 
def rt_score(true_value, predicted_value, max_error):
    error = abs(true_value - predicted_value)
    k = 1
    score = 1 / (1 + np.exp(k * (error / max_error-1)))
    
    return score
def pred_bde(mol,frags,mode='+'):
    import pickle  
    # load the trained model
    if mode == "+":
        with open('model/bde_predict_model_POS.pkl', 'rb') as f:  
            model = pickle.load(f) 
    else:
        with open('model/bde_predict_model_NEG.pkl', 'rb') as f:  
            model = pickle.load(f) 
    feats = []
    for j in range(frags.shape[0]):
        if isinstance(frags['bonds'][j],str):
            bids = eval(frags['bonds'][j])
        elif isinstance(frags['bonds'][j],list):
            bids = frags['bonds'][j]
        if not len(bids) > 0:
            bids = [0]
        bond_feat = bond_encoding(mol,bids)
        feats.append(bond_feat)
    
    bde = model.predict(feats)
    frags = frags.copy()
    frags['bde'] = bde
    return frags

def getmatched(truems2,predms2,mserror=20e-3,adduct='H'):
    """
    truemes and predms2 are datatables with atleast a column 'mz' and a columns 'intensity  '
    """
    if adduct == 'Na':
        truems2['mz'] = truems2['mz'] + 1.007825 - 22.989770
    elif adduct == 'Cl':
        truems2['mz'] = truems2['mz'] - 1.007825 - 34.968853
    elif adduct == 'K':
        truems2['mz'] = truems2['mz'] + 1.007825 - 38.963708

    truems2 = truems2.copy()
    truems2['intensity'] = truems2['intensity']/truems2['intensity'].sum()
    if not 'mid' in list(predms2.keys()):
        mid = [0]*len(predms2)
    else:
        mid = predms2['mid']       
            
    got_ms2 = []
    for m,b,_ in zip(predms2['mz'],predms2['bonds'],mid):
        delta = abs(truems2['mz'] - m)
        truems2['mserror'] = delta
        if (delta < mserror).any():
            temp = truems2[truems2['mserror'] < mserror].copy()
            temp['bonds'] = str(b)
            temp['mid'] = _
            got_ms2.append(temp)
            
    if len(got_ms2)>0:
        got_ms2 = pd.concat(got_ms2)
        got_ms2['mz'] = got_ms2['mz'].round(3)
        got_ms2 = got_ms2.drop_duplicates(subset=['mz']).reset_index(drop=True)
        mes = sum(np.e**(-0.5*(got_ms2['mserror']/mserror)**2))/len(truems2)
        got_ms2['mes'] = mes
    return got_ms2


def assign_nl(mz_table,mode,delta=20e-3):
    """
    Parameters
    ----------
    mz_table : table of mz and intensity, ranked inversely
        DESCRIPTION.
    mode : TYPE
        DESCRIPTION.
    delta : TYPE, optional
        DESCRIPTION. The default is 10e-3.
    Returns
    -------
    res : neutral loss table for graph plot.
    """
    # mz_table = pd.DataFrame(mz_table).copy()
    # mz_table.columns = ['mz','intensity']
    mz_table = mz_table.sort_values(by='mz',ascending=False).reset_index(drop=True) 
    nltable = pd.read_excel(r'../Source/available_nl.xlsx',sheet_name='msnl')
    # if mode == '-':
    #     nltable = nltable[nltable['Neg'].astype(bool)]
    # elif mode == '+':
    #     nltable = nltable[nltable['Pos'].astype(bool)]
    nl_ms = nltable['Accurate Mass']
    nls = nltable['Neutral Loss']
    nl_formula = nltable['Fragment formula']
    nl_smiles = nltable['Fragment SMILES']
    source_node = []
    target_node = []
    loss = []
    formula = []
    smiles = []
    s_id = []
    t_id = []
    s_int = []
    t_int = []
    for i in range(len(mz_table)):
        for j in range(i,len(mz_table)):
            source = mz_table['mz'][i]
            target = mz_table['mz'][j]
            ms_diff = source - target
            candidate_loss = [abs(ms_diff-i)<delta for i in nl_ms]
            if True in candidate_loss:
                source_node.append(source)
                target_node.append(target)
                loss.append(list(nls[candidate_loss])[0])
                formula.append(list(nl_formula[candidate_loss])[0])
                smiles.append(list(nl_smiles[candidate_loss])[0])
                s_id.append(mz_table['ID'][i])
                t_id.append(mz_table['ID'][j])
                s_int.append(mz_table['intensity'][i])
                t_int.append(mz_table['intensity'][j])
    res = pd.DataFrame([s_id,t_id,source_node,target_node,s_int,t_int,
                        loss,formula,smiles]).T
    res.columns = ['Source_ID','Target_ID','Source','Target','SourceInt','TargetInt',
                   'Neutral Loss','Formula','SMILES']
    return res

def assign_nl2(mz_table,mode,delta=10e-3):
    mz_table = mz_table.sort_values(by='mz',ascending=False).reset_index(drop=True) 
    nltable = pd.read_excel(r'Source/available_nl.xlsx',sheet_name='msnl')
    nltable = nltable[nltable['Frequency']>=1]
    nl_ms = nltable['Accurate Mass']
    # nls = nltable['Neutral Loss']
    # nl_formula = nltable['Fragment formula']
    # nl_smiles = nltable['Fragment SMILES']
    s_id = []
    t_id = []
    nloss = []
    for i in range(len(mz_table)):
        for j in range(i,len(mz_table)):
            source = mz_table['mz'][i]
            target = mz_table['mz'][j]
            ms_diff = source - target
            candidate_loss = nltable[[abs(ms_diff-i)<delta for i in nl_ms]].reset_index(drop=True)
            if len(candidate_loss)>0:
                s_id.append(mz_table['ID'][i])
                t_id.append(mz_table['ID'][j])
                nloss.append(candidate_loss['Fragment formula'][0])
    res = pd.DataFrame([s_id,t_id,nloss]).T
    res.columns = ['Source_ID','Target_ID','nloss']
    return (mz_table,res)

      
def one_of_k_encoding_unk(x,allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s:x == s,allowable_set))

def get_atom_features(atom):
    possible_atom = ['C','H','O','N','P','S','Un'] # Un represents the other atoms
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(),possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(),[0,1,2,3,4])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(),[0,1,2,3,4,5,6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(),[-1,0,1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
                                          [Chem.rdchem.HybridizationType.SP,
                                          Chem.rdchem.HybridizationType.SP2,
                                          Chem.rdchem.HybridizationType.SP3,
                                          Chem.rdchem.HybridizationType.SP3D])
    return np.array(atom_features)
    
def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats,dtype=np.int32)

def bond_encoding(mol,bids):
  #  fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) 
    fp = np.array(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()),dtype=int)
    
    if not isinstance(bids,list):
        bids = [bids]
 
    all_bond_feat = []
    for bid in bids:
        bond = mol.GetBondWithIdx(bid)
        bond_feat = get_bond_features(bond)
        b_at = bond.GetBeginAtom()
        e_at = bond.GetEndAtom()
        #b_neib = [a.GetIdx() for a in b_at.GetNeighbors()]
        #b_neib.remove(e_at.GetIdx())
        #e_neib = [a.GetIdx() for a in e_at.GetNeighbors()]
        #e_neib.remove(b_at.GetIdx())
        #b_bonds_of_neib = [mol.GetBondBetweenAtoms(b,b_at.GetIdx()).GetBondTypeAsDouble() for b in b_neib]
        #e_bonds_of_neib = [mol.GetBondBetweenAtoms(b,e_at.GetIdx()).GetBondTypeAsDouble() for b in e_neib]
        b_at_feat = get_atom_features(b_at)
        e_at_feat = get_atom_features(e_at)
        _feat = np.concatenate([bond_feat,b_at_feat,e_at_feat],dtype=int)
        all_bond_feat.append(_feat)
    all_bond_feat = np.vstack(all_bond_feat)
    all_bond_feat = all_bond_feat.sum(axis=0)
    feat = np.concatenate([fp,all_bond_feat])
    return feat

class Node:
    def __init__(self, key, info):
        self.key = key
        self.info = info
        self.children = []

    def to_dict(self):
        result = dict()
        result["key"] = self.key

        for _k, _v in self.info.items():
            result[_k] = _v
        result["children"] = [x.to_dict() for x in self.children]
        return result
    
class TreeProcessor:
    def __init__(self, pair: list, info: dict, d_type="json"):
        """

        :param pair: [(0, 1), (0, 2)...] parent-child pair
        :param info: {0 : {"name": xxx}} detailed information for each node
        :param d_type: return type json dict
        """
        self._pair = pair
        self._info = info
        self.d_type = d_type
        self.root_id, self._pair_map, self._ids = self._process_data()

    def _process_data(self):
        """
        transfer [(0, 1), (0, 2)...] parent-child pair into {father: [child, child]}
        :return:
        """
        pair_map = defaultdict(list)
        ids = set()
        fathers = set()
        children = set()
        for f, s in self._pair:
            ids.add(f)
            ids.add(s)
            fathers.add(f)
            children.add(s)
            pair_map[f].append(s)
        root_id = fathers - children
        root_id = list(root_id)[0] if root_id else None
        return root_id, pair_map, ids

    def _create_nodes(self):
        return {x: Node(x, self._info.get(x, {})) for x in self._ids}

    @property
    def tree(self):
        node_map = self._create_nodes()
        for k, v in self._pair_map.items():
            node_map[k].children = [node_map[i] for i in v]
        root_node: Node = node_map.get(self.root_id)

        tree_res = json.dumps(root_node.to_dict()) if self.d_type == 'json' else root_node.to_dict()
        return tree_res
    



def trimNodes(df,links = []):
    parents = df['Source_ID']
    children = df['Target_ID']
    for i in range(len(parents)):
        onelink = [parents[i]]
        sub = df[parents == children[i]].reset_index(drop=True)
        if len(sub)>0:
            sublink = trimNodes(sub,links=onelink)
            onelink.append(sublink)
        else:
            onelink.append(children[i])
            links.append(onelink)
    return links
