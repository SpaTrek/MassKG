from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors
import numpy as np
import pandas as pd
import re
import itertools
from rdkit.Chem import MACCSkeys




        
        


def pop_list(l):
    out = []
    for x in l :
        
        if  len(out)<1:
            out.append(x)
        else:
            if not x in out:
                out.append(x)
    return out

def pred_bde(mol,frags,mode='+'):
    import pickle  
    # load the trained model
    if mode == "+":
        with open(f'/slurm/home/yrd/liaolab/zhubingjie/MassKG/bde_predict_model_POS.pkl', 'rb') as f:  
            model = pickle.load(f) 
    else:
        with open(f'/slurm/home/yrd/liaolab/zhubingjie/MassKG/bde_predict_model.pkl', 'rb') as f:  
            model = pickle.load(f) 
    feats = []
    for j in range(frags.shape[0]):
        bids = eval(frags['bonds'][j])
        if not len(bids) > 0:
            bids = [0]
        bond_feat = bond_encoding(mol,bids)
        feats.append(bond_feat)
    
    bde = model.predict(feats)
    frags = frags.copy()
    frags['bde'] = bde
    return frags
        
        

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


def get_fragments0(mol,bonds_comb,adduct,mode):
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
        fmol = Chem.FragmentOnBonds(mol,bd) #断裂键
        try:
            frag_mols = Chem.GetMolFrags(fmol,asMols=True) #提取碎片分子(剔除键后) 
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
        n_ids = [n if n else [0] for n in n_ids] # 似乎0号键不会被记录,这里补上
        # 注意，这里是反的，O-连的是C所以O反而会是-1，所以乘以-1改回来
        n_vals = [[-1*val_dict[s] for s in n] for n in n_ids]
        n_breaks = [len(re.findall("-.\d*#0",s))+ 2*len(re.findall("=.\d*#0",s)) for s in frag_smarts]
        # n_breaks = [min(a,b) for a,b in zip(n_Hs,n_breaks)] # 取自由基与活泼H的最小值
        n_atoms = [i.GetNumAtoms() for i in frag_mols]
        # 加减H规则，不同环境可能的方法不同，但要求所有碎片总共加减H为0
        fw = []
        ff = []
        ab = []
        for i in range(len(frag_mols)):
            if n_charges[i] == 0: #判断是否带电
                continue
            
            a = n_breaks[i]
            vals = n_vals[i]
            vals = [int(v/abs(v)) if v!=0 else int(v) for v in vals]
            if n_atoms[i] > 2: 
                if n_Hs[i] > 0: 
                    b = [0] + [-1*(j+1) for j in range(a)] + [(j+1) for j in range(a)]
                else:
                    b = [0] + [(j+1) for j in range(a)] #对于缺少H的基团只能得到H而无法给出
                if min(n_atoms)>2: # 小分子特权
                    for v in vals:
                        if v == -1:
                            b.remove(max(b))
                        elif v == 1:
                            b.remove(min(b))
            else:
                # 小分子特权,只拿不给，如CH3 HO NH2等
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
        # ab_是不同碎片间所有可能的组合
        for a_b_ in ab_:
            if not sum(a_b_) == 0 :
                continue
            fw_ = [rdMolDescriptors.CalcExactMolWt(frag_mols[i])+a_b_[i]*H for i in range(len(a_b_))]
            # ff_ = [(frag_mols[i],'{}H'.format(i)) for i in range(len(a_b_))] # 暂时不返回碎片
            ff_ = [frag_mols[i]for i in range(len(a_b_))]
            ff_ = [Chem.MolToSmarts(f) for f in ff_]#删除断裂的键
            ff_ = [s.replace(p,"") for s in ff_ for p in  re.findall("..\d*#0.",s)]
            if not max(fw_)>=50:
                continue
            fw.extend(fw_)
            ff.extend(ff_)
        fragments.append(ff)
        frag_weights.append(fw)
        all_bonds.append(bd)
    frag_weights.append([rdMolDescriptors.CalcExactMolWt(mol)]) #母离子加进去)
    fragments.append([Chem.MolToSmarts(mol)])
    all_bonds.append([])
    if mode == '-':
        frag_weights = [[f - adct for f in fw] for fw in frag_weights]
    elif mode == '+':
        frag_weights = [[f + adct for f in fw] for fw in frag_weights]
    return fragments,frag_weights,all_bonds


def get_fragments(mol, bonds_comb, adduct, mode):
    # 常量定义
    mass_dict = {
        'H': 1.007825,
        'Na': 22.98977,
        'K': 38.963707,
        'Cl': 34.968853
    }
    
    adct = mass_dict.get(adduct, mass_dict['H']) * (-1 if 'Cl' in adduct else 1)

    fragments = []
    frag_weights = []
    all_bonds = []
    
    symb_val_rank = {'C': 0, 'O': 2, 'N': 1, 'P': 0, 'S': 0, 'na': 0}

    for bd in bonds_comb:
        if not isinstance(bd, list):
            bd = [bd]
        if not bd:
            continue

        bd = list(set(bd))
        bateat = [(mol.GetBondWithIdx(b).GetBeginAtom(), mol.GetBondWithIdx(b).GetEndAtom()) for b in bd]
        atoms_idx = [[a.GetIdx() for a in be] for be in bateat]
        atoms_symb = [[a.GetSymbol() for a in be] for be in bateat]
        atoms_symb = [[a if a in symb_val_rank else 'na' for a in be] for be in bateat]
        atoms_ms = [[a.GetMass() for a in be] for be in bateat]
        
        # 计算原子间的值
        atoms_val = [[symb_val_rank[be[0]] - symb_val_rank[be[1]], symb_val_rank[be[1]] - symb_val_rank[be[0]]] for be in atoms_symb]
        val_dict = {a: b for aa, bb in zip(atoms_idx, atoms_val) for a, b in zip(aa, bb)}
        
        fmol = Chem.FragmentOnBonds(mol, bd)
        
        try:
            frag_mols = Chem.GetMolFrags(fmol, asMols=True)
        except Exception as e:
            print(e)
            continue
        
        frag_smarts = [Chem.MolToSmarts(f) for f in frag_mols]
        
        n_charges = [rdMolDescriptors.CalcNumHeteroatoms(mol) if mode == '-' else 1 for f in frag_mols]
        n_Hs = [sum(a.GetNumImplicitHs() for a in f.GetAtoms()) for f in frag_mols]
        n_ids = [[eval(s.replace('#0', '')) for s in re.findall(r"\d+#0", s)] for s in frag_smarts]
        n_ids = [n if n else [0] for n in n_ids]
        
        n_vals = [[-val_dict[s] for s in n] for n in n_ids]
        n_breaks = [len(re.findall(r"-.\d*#0", s)) + 2 * len(re.findall(r"=.\d*#0", s)) for s in frag_smarts]
        n_atoms = [i.GetNumAtoms() for i in frag_mols]
        
        ab = []
        for i in range(len(frag_mols)):
            if n_charges[i] == 0:
                continue
            
            a = n_breaks[i]
            vals = [int(v / abs(v)) if v != 0 else int(v) for v in n_vals[i]]
            b = [0] + ([-1 * (j + 1) for j in range(a)] + [(j + 1) for j in range(a)]) if n_Hs[i] > 0 else [0] + [(j + 1) for j in range(a)]
            
            if min(n_atoms) > 2:
                for v in vals:
                    if v == -1:
                        b.remove(max(b))
                    elif v == 1:
                        b.remove(min(b))
            ab.append(b)

        if len(ab) < 2:
            continue
        
        ab_ = itertools.product(*ab)

        for a_b_ in ab_:
            if sum(a_b_) != 0:
                continue
            
            fw_ = [rdMolDescriptors.CalcExactMolWt(frag_mols[i]) + a_b_[i] * mass_dict['H'] for i in range(len(a_b_))]
            ff_ = [Chem.MolToSmarts(frag_mols[i]) for i in range(len(a_b_))]
            ff_ = [s.replace(p, "") for s in ff_ for p in re.findall(r"..\d*#0.", s)]
            
            if max(fw_) < 50:
                continue
            
            frag_weights.append(fw_)
            fragments.append(ff_)
        
    frag_weights.append([rdMolDescriptors.CalcExactMolWt(mol)])
    fragments.append([Chem.MolToSmarts(mol)])
    all_bonds.append([])

    frag_weights = [[f + adct if mode == '+' else f - adct for f in fw] for fw in frag_weights]

    return fragments, frag_weights, all_bonds




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


def break_all_single(mol,mode='-',adduct=''):
    """
    断开化合物的全部单键
    """
    fragments = []
    frag_weigths = []
    all_bonds = [] 
    # 获取所有脂肪单键
    chain_bonds = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing() and b.GetBondTypeAsDouble()<=2] 
    bonds_comb = chain_bonds
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
    fragments,frag_weigths,all_bonds = get_fragments(mol,bonds_comb,adduct=adduct,mode=mode)
    return fragments,frag_weigths,all_bonds

def break_all2(mol,mode='-',adduct=''):
    fragments = []
    frag_weigths = []
    all_bonds = []
    ri = mol.GetRingInfo() # ring information
    bonds_in_r = ri.BondRings()
    bridge_bonds = get_bridge_bonds(bonds_in_r) #获取桥键，不断裂
    bonds_in_r = [[b_ for b_ in b if b_ not in bridge_bonds] for b in bonds_in_r]
    
    chain_bonds = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing() and b.GetBondTypeAsDouble()<=2]
    ring_bonds = [[[xza[i],xza[j]] for i in range(len(xza)) for j in range(i,len(xza)) if i!=j] for xza in bonds_in_r] # 换上的键需要成组断
    
    chain_comb = bondscomb2(chain_bonds,chain_bonds) # 单键+单键组合
    ring_comb = [bondscomb2(ring_bonds[i],ring_bonds[j]) for i in range(len(ring_bonds)) for j in range(i,len(ring_bonds)) if i!=j] # 两个环的组合
    ri_ch_comb = [bondscomb2(chain_bonds,i) for i in ring_bonds] # 生成单键+环2键组合 
    
    
    # 整理键组
    ring_bonds = [b for bs in ring_bonds for b in bs] 
    ring_comb = [b for bs in ring_comb for b in bs] 
    ri_ch_comb = [b for bs in ri_ch_comb for b in bs]
    bonds_comb = chain_bonds + ring_bonds + chain_comb + ring_comb + ri_ch_comb
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
    bonds_comb = pop_list(bonds_comb)
    fragments,frag_weigths,all_bonds = get_fragments(mol,bonds_comb,adduct=adduct,mode=mode)
    return fragments,frag_weigths,all_bonds

def break_KG2(mol,bonds,mode='-',adduct=''):
    '''
    这个是用来做预测结果应用的

    '''
    fragments = []
    frag_weigths = []
    all_bonds = []
    ri = mol.GetRingInfo() # ring information
    bonds_in_r = ri.BondRings()
    bridge_bonds = get_bridge_bonds(bonds_in_r) #获取桥键，不断裂
    bonds_in_r = [[b_ for b_ in b if b_ not in bridge_bonds and b_ in bonds] for b in bonds_in_r]
    
    chain_bonds = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing()]
    chain_bonds = [b for b in chain_bonds if b in bonds]
    ring_bonds = [[[xza[i],xza[j]] for i in range(len(xza)) for j in range(i,len(xza)) if i!=j] for xza in bonds_in_r if len(xza)>1] # 换上的键需要成组断
    
    chain_comb = bondscomb2(chain_bonds,chain_bonds) # 单键+单键组合
    ring_comb = [bondscomb2(ring_bonds[i],ring_bonds[j]) for i in range(len(ring_bonds)) for j in range(i,len(ring_bonds)) if i!=j] # 两个环的组合
    ri_ch_comb = [bondscomb2(chain_bonds,i) for i in ring_bonds] # 生成单键+环2键组合 
    
    
    # 整理键组
    ring_bonds = [b for bs in ring_bonds for b in bs] 
    ring_comb = [b for bs in ring_comb for b in bs] 
    ri_ch_comb = [b for bs in ri_ch_comb for b in bs]
    bonds_comb = chain_bonds + ring_bonds + chain_comb + ring_comb + ri_ch_comb
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
    bonds_comb = pop_list(bonds_comb)
    fragments,frag_weigths,all_bonds = get_fragments(mol,bonds_comb,adduct=adduct,mode=mode)
    return fragments,frag_weigths,all_bonds


def break_t32(mol,mode='-',adduct=''):
    patterns = {
    'N': AllChem.MolFromSmarts('[#7]'),          
    'O': AllChem.MolFromSmarts('[OD2;!R]'), # 非环醚键
    'P': AllChem.MolFromSmarts('[#15;!R]'),
    'S': AllChem.MolFromSmarts('[#16;!R]'),
    'CO': AllChem.MolFromSmarts('[CX3]=O'),
    }
    patterns2 = {
        'OH': AllChem.MolFromSmarts('[OX2H1]'),
        'CH3': AllChem.MolFromSmarts('[CH3]'),
        'C':AllChem.MolFromSmarts('[#6D3]'),
        }
    patterns3 = {'C': AllChem.MolFromSmarts('[#6D3R]-[#6R]')} # 叔C
    ring_za = {'Cr6': AllChem.MolFromSmarts('[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1'), # 6元C环
               'Or6': AllChem.MolFromSmarts('[#6]1~[#6]~[#6]~[#6](=[#8])~[#8]~[#6]~1'),
               'Or5': AllChem.MolFromSmarts('[#6]1~[#6]~[#6](=[#8])~[#8]~[#6]~1'),
               'CO': AllChem.MolFromSmarts('[CD3]=O'),
               }
    resonance = {'xc1':AllChem.MolFromSmarts('[#6H2R]-[CX3R](=O)'),
                 'xc2':AllChem.MolFromSmarts('[#6](=O)-[#6]-[#6](=O)'),
        }
    fragments = []
    frag_weigths = []
    all_bonds = []
    # mol =  Chem.RWMol(mol)
    ri = mol.GetRingInfo() # ring information
    bonds_in_r = ri.BondRings()
    bridge_bonds = get_bridge_bonds(bonds_in_r) #获取桥键，不断裂
    x1 = [mol.GetSubstructMatches(p) for k,p in patterns.items()] #全部匹配
    x2 = [mol.GetSubstructMatch(p) for k,p in patterns2.items()] #特定官能团只匹配一次
    x3 = [mol.GetSubstructMatches(p) for k,p in patterns3.items()] # 环间叔C
    xrza = [mol.GetSubstructMatches(p) for k,p in ring_za.items()] # 所有杂环
    xres = [mol.GetSubstructMatch(p) for k,p in resonance.items()] # 烯醇式共振
    x = [a for b in x1+x2 for a in b]
    xb = [a for b in x3 for a in b]
    xrza_ = [a for b in xrza for a in b]
    xres_ = [a for b in xres for a in b ]
    x = [a if type(a)==int else a[0] for a in x]
    # xrza_ = [[a for a in b if mol.GetAtomWithIdx(a).GetNumImplicitHs()<=1] for b in xrza_] #杂环叔C 
    x = set(x)
    atoms1 = [mol.GetAtomWithIdx(i) for i in x]
    bonds1 = [at.GetBonds() for at in atoms1]
    bonds1 = [b for bs in bonds1 for b in bs]
    bonds1 = [b for b in bonds1 if not b.IsInRing()] #不断环上键
    bonds1 = [b for b in bonds1 if b.GetBondTypeAsDouble()==1] # 只断单键
    bonds3 = [mol.GetBondBetweenAtoms(b[0],b[1]) for b in xb]
    bonds3 = [b for b in bonds3 if not b.IsInRing()] # 非环C-C
    bonds4 = [[mol.GetBondBetweenAtoms(at,bt) for at in a for bt in a] for a in xrza_]
    bonds4 = [[b for b in bs if b] for bs in bonds4]
    bonds4 = [[b for b in bs if b.GetBeginAtom().GetNumImplicitHs()+b.GetEndAtom().GetNumImplicitHs()<=3] for bs in bonds4]
    bonds4_ = []
    atoms5 = [mol.GetAtomWithIdx(i) for i in xres_]
    atoms5 = [a for a in atoms5 if a.GetSymbol()=='O']
    bonds5 = [a.GetBonds() for a in atoms5]
    for b in bonds4:
        bonds4_.append([x for x in b if x])
    bonds4_ = [[b for b  in bs if b.IsInRing()] for bs in bonds4_] # 杂环键
    bonds4_ = [[b for b  in bs if b.GetBondTypeAsDouble()==1] for bs in bonds4_] # 杂环键
    idx = [b.GetIdx() for b in bonds1+bonds3] # 获得键ID
    idx_rza = [[b.GetIdx() for b in bs] for bs in bonds4_] 
    idx_rza = [[b for b in bs if not b in bridge_bonds] for bs in idx_rza] # 不断桥键
    idx_res = [b.GetIdx() for bs in bonds5 for b in bs] 
    
    idx2_2 = []
    idx_rza_comb = []
    idx = list(set(idx))
    idx2_1 = bondscomb2(idx,idx) # 单键+单键组合
    if len(idx_rza)>0:
        idx_rza = [list(set(i)) for i in idx_rza if len(i)>0]
        idx_rza = pop_list(idx_rza)
        # idx_rza = [[[xza[i],xza[j]] for i in range(len(xza)) for j in range(i,len(xza)) if i!=j] for xza in idx_rza] # 环上的键需要成组断
        idx_rza = [bondscomb2(i,i) for i in idx_rza]
        idx_rza_comb = [bondscomb2(idx_rza[i],idx_rza[j]) for i in range(len(idx_rza)) for j in range(i,len(idx_rza)) if i!=j] # 两个环的组合
        idx2_2 = [bondscomb2(idx,i) for i in idx_rza ] # 生成单键+环2键组合
        
        idx_rza = [j for i in idx_rza for j in i]
        idx_rza_comb = [j for i in idx_rza_comb for j in i] # 整合2键的组合
        idx2_2 = [j for i in idx2_2 for j in i]
        # 如果idx_rza 有多个环的话需要将相关的bonds组合拼接起来
            
    idx2 = idx2_1+idx2_2 # 整合有单键的组合
    bonds_comb = idx + idx_rza + idx_rza_comb + idx2 + idx_res
    bonds_comb = pop_list(bonds_comb)
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
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
        cent_id = list([c for c in cent_id if len(c)==2][0]) # only the para atoms  
        
        comb1 = [atnebs[0][0],cent_id[0],atnebs[0][1]]
        comb2 = [atnebs[1][0],cent_id[1],atnebs[1][1]]
        
        
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
    
     
def fragments_generation(smiles,mode='',t=None):
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

    if t == 'ht':
        try:
            f,mids,fw,bs = multi_break2(mol,adduct=adduct,mode=mode)
            mids = [0]*len(fw)
            # mol_smiles = [sms]*len(fw)
        except Exception as e:
            sick_mols.append(mol)
            temp_ = pd.DataFrame({'mz':[0],'smarts':[0],'bonds':[0],'BDE':[120],'mid':[0]})
            return temp_
            
    else:
        try:
            f,fw,bs = break_all2(mol,adduct=adduct,mode=mode)
            mids = [0]*len(fw)
        except Exception as e:
            sick_mols.append(mol)
            temp_ = pd.DataFrame({'mz':[0],'smarts':[0],'bonds':[0],'BDE':[120],'mid':[0]})
            return temp_
        
    mz = []
    fragsmts = []
    BDEs = []
    bonds = []
    molids = []
    for f_,m_,b_,i_ in zip(f,fw,bs,mids):
        for _1,_2 in zip(f_,m_):
            fragsmts.append(_1)
            mz.append(np.round(_2,3))
            bonds.append(b_)
            BDEs.append(0)
            molids.append(i_)
    
    temp_ = pd.DataFrame([mz,fragsmts,bonds,BDEs,molids]).T 
    temp_.columns = ['mz','smarts','bonds','BDE','mid'] 
    temp_ = temp_.sort_values(by='BDE',ascending=True)
    temp_ = temp_.drop_duplicates(subset=['mz']).dropna()
    temp_ = temp_.reset_index(drop=True)
    return temp_

def break_ht2(mol,mode='-',adduct='',subclass='',CH3=False):
    patterns = {
    'N': AllChem.MolFromSmarts('[#7]'),          
    'O': AllChem.MolFromSmarts('[OD2;!R]'), # 非环醚键
    'P': AllChem.MolFromSmarts('[#15;!R]'),
    'S': AllChem.MolFromSmarts('[#16;!R]'),
    'CO': AllChem.MolFromSmarts('[CX3]=O'), # 脂肪CO
    }
    patterns1 = {
    'R-R': AllChem.MolFromSmarts('[#6]1(~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]~2)~[#6]~[#6]~[#6]~[#6]~[#6]~1'), # 联苯
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
    patterns3 = {'C': AllChem.MolFromSmarts('[#6D3R]-[#6R]')} # 环间叔C
    patterns4 = {'char2':AllChem.MolFromSmarts('[#8]=[#6](-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1)/[#6]=[#6]/[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'),
        }
    ring_zares = {'cch':AllChem.MolFromSmarts('[#6;!R]=[#6;!R]-[#8H;!R]')
        }
    ring_za = {'Or6': AllChem.MolFromSmarts('[#6]1~[#6]~[#6]~[#6]~[#6]~[#8]~1'), # 环醚键
               'Or5': AllChem.MolFromSmarts('[#6]1~[#6]~[#6]~[#6]~[#8]~1'), # 环醚键
               'CO': AllChem.MolFromSmarts('[#6D3R]=[#8]'), #环羰基
               }

    ring_res = {'xch':AllChem.MolFromSmarts('[#6]1=[#6](-[#8H])-[#6]=[#6]-[#6]=[#6]-1'), # 环上烯醇
                'xcm':AllChem.MolFromSmarts('[#8D2;!R]-[#6]1-[#6]=[#6](-[#8D2;!R])-[#6]=[#6]-[#6]=1'), # 烯醚
                'RDA':AllChem.MolFromSmarts('[#6]1-[#6]-[#6]-[#6]=[#6](-[#8H])[#8]-1'),
                'RDA2':AllChem.MolFromSmarts('[#6]1-[#6]-[#6]-[#6](-[#8H])=[#6]-[#8]-1'),
                }
    fragments = []
    frag_weigths = []
    all_bonds = []
    # mol =  Chem.RWMol(mol)
    ri = mol.GetRingInfo() # ring information
    bonds_in_r = ri.BondRings()
    bridge_bonds = get_bridge_bonds(bonds_in_r) #获取桥键，不断裂
    
    
    x1 = [mol.GetSubstructMatches(p) for k,p in patterns.items()] #全部匹配
    x1_ = [mol.GetSubstructMatches(p) for k,p in patterns1.items()] # 环间的特定C
    x2 = [mol.GetSubstructMatches(p) for k,p in patterns2.items()] #特定官能团只匹配2次
    x2 = [a[:2] for a in x2]
    x3 = [mol.GetSubstructMatches(p) for k,p in patterns3.items()] # 环间叔C
    x4 = [mol.GetSubstructMatches(p) for k,p in patterns4.items()]
    xrzares = [mol.GetSubstructMatches(p) for k,p in ring_zares.items()]
    xrza = [mol.GetSubstructMatches(p) for k,p in ring_za.items()] # 所有杂环
    xresr = [mol.GetSubstructMatches(p) for k,p in ring_res.items()] # 苯酚共振
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
    xrzares_ = set(xrzares_) # 没用
    
    atoms1 = [mol.GetAtomWithIdx(i) for i in x]
    bonds1 = [at.GetBonds() for at in atoms1]
    bonds1 = [b for bs in bonds1 for b in bs]
    bonds1 = [b for b in bonds1 if not b.IsInRing()] #不断环上键
    bonds1 = [b for b in bonds1 if b.GetBondTypeAsDouble()==1] # 只断单键
    bonds3 = [mol.GetBondBetweenAtoms(b[0],b[1]) for b in xb]
    bonds3 = [b for b in bonds3 if not b.IsInRing()] # 非环C-C
    atoms3 = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in bonds3]
    atoms3 = [a for at in atoms3 for a in at] #获取环间叔C编号用于下面识别有叔C的杂环裂解
    
    # 下面是环上键
    xrza__ = []
    for xr in xrza_:
        if len(xr) > 4:# 连有叔碳的杂环才可碎裂
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
    bonds5_0 = [[mol.GetBondBetweenAtoms(at,bt) for at in a for bt in a] for a in xresr_]# 苯酚共振
    bonds5_0 = [[b for b in bs if b] for bs in bonds5_0]
    # bonds5_0 = [[b for b in bs if b.GetBondTypeAsDouble()==1] for bs in bonds5_0]
    bonds5 = [[a.GetBonds() for a in at] for at in atoms5]
    bonds5 = [[b_  for b in bs for b_ in b]for bs in bonds5]
    bonds5 = [b0+b1 for b0,b1 in zip(bonds5_0,bonds5)] # 单键+符合的双键
    bonds5 = [[b for b in bs if b.IsInRing()] for bs in bonds5]#
    atoms6 = [mol.GetAtomWithIdx(i) for i in x4] # 俩都是双键应该可以合并
    bonds6 = [at.GetBonds() for at in atoms6]
    bonds6 = [b for bs in bonds6 for b in bs]
    bonds6 = [b for b in bonds6 if not b.IsInRing()] #不断环上键
    bonds6 = [b for b in bonds6 if b.GetBondTypeAsDouble()==2] # 只断双键
    # 整合b4 b5
    bonds4_ = []
    for b in bonds4+bonds5: # 合并环上键对剔除空值
        bonds4_.append([x for x in b if x])
    # 获得键ID
    # idx = [b.GetIdx() for b in bonds1+bonds3+bonds6]
    idx = [b.GetIdx() for b in bonds1+bonds6] # 非环单双键[1.删除bonds3]
    idx = list(set(idx))

    idx_rza = [[b.GetIdx() for b in bs] for bs in bonds4_] 
    idx_rza = [[b for b in bs if not b in bridge_bonds] for bs in idx_rza] # 不断桥键
    
    idx2_2 = []
    idx_rza_comb = []
    idx = list(set(idx))
    idx2_1 = bondscomb2(idx,idx) # 单键+单键组合
    if len(idx_rza)>0:
        idx_rza = [list(set(i)) for i in idx_rza if len(i)>0]
        idx_rza = pop_list(idx_rza)
        # idx_rza = [[[xza[i],xza[j]] for i in range(len(xza)) for j in range(i,len(xza)) if i!=j] for xza in idx_rza] # 换上的键需要成组断
        idx_rza = [bondscomb2(i,i) for i in idx_rza]
        idx_rza_comb = [bondscomb2(idx_rza[i],idx_rza[j]) for i in range(len(idx_rza)) for j in range(i,len(idx_rza)) if i!=j] # 两个环的组合
        idx2_2 = [bondscomb2(idx,i) for i in idx_rza ] # 生成单键+环2键组合
        # idx_rza_comb = [j for i in idx_rza_comb for j in i] # 整合2键的组合
        # 如果idx_rza 有多个环的话需要将相关的bonds组合拼接起来
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
            
    idx2 = idx2_1+idx2_2 # 整合有单键的组合
    bonds_comb = idx + idx_rza + idx_rza_comb + idx2
    bonds_comb = pop_list(bonds_comb)
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
    fragments,frag_weigths,all_bonds = get_fragments(mol,bonds_comb,adduct=adduct,mode=mode)
    return fragments,frag_weigths,all_bonds