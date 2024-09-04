from tqdm import tqdm, trange
from msp_writer import parse_MSDIAL,msptranslator
from mgf_translator import mgftranslator
import numpy as np
import pandas as pd
from insiliconFRAG import *
from calcformula import *
import os
import re
from collections import defaultdict

def API(project_path,k,thresh,dbparams,mode='+'):
    # show_cols = ['ID','Rt','Mz','Size','exact_mass','Formula','adduct','MassKGID',
    # 'InChIKey','SMILES','Chemical Name','Class','Subclass','onto','final_score','level']
    if mode == '-':
        subpath = 'NEG'
    elif mode == '+':
        subpath = 'POS'
    ppm = 10
    mDa = 5
    formuladb = dbparams[3]
    # tree = build_tree(mode=mode)
    datapath = os.path.join(project_path,'MassKG_test_INPUTS',subpath)
    outputpath = os.path.join(project_path,'MassKG_test_OUTPUTS')
    fragpath = os.path.join(project_path,'insilicoFRAGs')
    assert os.path.isdir(datapath),'Please select a folder'
    f_names = os.listdir(datapath)
    for f in f_names:        
        pepmass,test_ms,feature_table = read_file(datapath,f,thresh)
        print(f.split('.')[0],' Start Analyzing... ...' )
        temp_path = os.path.join(outputpath,f.split('.')[0])
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        if not "ID" in feature_table:
            feature_table['ID'] = range(feature_table.shape[0])
        name_ids = feature_table['ID']

        if "Adduct" in feature_table:
            adducts = list(feature_table['Adduct'])
            res = [get_formula(mode,m,formuladb,adduct=a,mDa=mDa,ppm=ppm) for m,a in zip(pepmass,adducts)]
        else:
            res = [get_formula(mode,m,formuladb,adduct=None,mDa=mDa,ppm=ppm) for m in pepmass]
        # ontores = assign_ontos(res,test_ms,mode=mode)
        # feature_table['onto'] = ontores
        summerize_table = []
        for i in tqdm(range(len(test_ms)),desc='Processing'):
            if not len(res[i])>0:
                continue
            x = get_rank(res[i],test_ms[i],mode,fragpath,dbparams) 
            if len(x) > 0 and len(x[0])>0:
                x_ = pd.concat([feature_table.iloc[i]]*len(x[0]),axis=1).T.reset_index(drop=True)
                x_0 = pd.concat([x_,x[0]],axis=1)
                rankid = [str(int(j))+'_'+str(i) for i,j in enumerate(list(x_0['ID']))]
                x_0.insert(0,'ID_Rank',rankid)
                x_1 = x[1]# matched fragments
                x_0 = x_0[[len(_)>0 for _ in x_1]] 
                x_1 = [_ for _ in x_1 if len(_)>0][:k]
                spectra = x_0.iloc[:k,:].copy()
                write_fragments = []
                for n in range(len(x_1)):
                    fragments = ''
                    interactions = x_1[n][1].copy()
                    infos = x_1[n][0].copy()
                    if not len(x_1[n][1]) > 0: 
                        write_fragments.append(fragments)
                        continue
                    fm_map = {a:b for a,b in zip(infos['ID'],infos['formula']) if a in list(interactions['Source_ID'])+list(interactions['Target_ID'])}
                    mz_map = {a:b for a,b in zip(infos['ID'],infos['mz']) if a in list(interactions['Source_ID'])+list(interactions['Target_ID'])}
                    
                    count = 0
                    while "" in fm_map.values() and count<len(interactions):
                        interactions['Source_formula'] = interactions['Source_ID'].map(fm_map)
                        interactions['Target_formula'] = interactions['Target_ID'].map(fm_map)
                        interactions['Source_mz'] = interactions['Source_ID'].map(mz_map)
                        interactions['Target_mz'] = interactions['Target_ID'].map(mz_map)
                        sfm = interactions['Source_formula']
                        nfm = interactions['nloss'] 
                        tfm = [formula_jis(fm1, fm2) if fm1 != "" else "" for fm1,fm2 in zip(sfm,nfm)]
                        tfm = [re.findall('[0-9A-Z]+', f) for f in tfm]
                        tfm = [f[0] if len(f)==1 else "" for f in tfm]
                        interactions['Target_formula'] = tfm
                        fm_map = {a:b for a,b in zip(list(interactions['Source_ID'])+list(interactions['Target_ID']),list(interactions['Source_formula'])+list(interactions['Target_formula']))}
                        count += 1
                    for jj in range(interactions.shape[0]):
                        fragments += str(interactions['Source_mz'][jj])
                        fragments += '['+interactions['Source_formula'][jj]+']' 
                        fragments += '-'+'['+interactions['nloss'][jj]+']'
                        fragments += str(interactions['Target_mz'][jj])
                        fragments += '['+interactions['Target_formula'][jj]+']'
                        fragments += '\n'
                    write_fragments.append(fragments)
                spectra['fragments'] = write_fragments
                summerize_table.append(spectra)
        summerize_table = pd.concat(summerize_table)
        summerize_table.to_csv(os.path.join(temp_path,'Summarysheet.csv'),index=False)
        print(f.split('.')[0],' Finished')
