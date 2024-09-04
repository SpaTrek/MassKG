import re
import pandas as pd
import bisect
from scipy.spatial.distance import cosine
from scipy import signal
import numpy as np
import gc
import json
import time
from collections import defaultdict



def isotope_pattern(formula,charge=0):
    from brainpy import isotopic_variants
    '''
    Task: 
        Generate theoretical isotope distribution
    ref:
        P. Dittwald, J. Claesen, T. Burzykowski, D. Valkenborg, and A. Gambin,
        “BRAIN: a universal tool for high-throughput calculations of the isotopic
        distribution for mass spectrometry.,” Anal. Chem., vol. 85, no. 4,
        pp. 1991–4, Feb. 2013.
    git:
        https://github.com/mobiusklein/brainpy
    '''
    formula = re_formula(formula)
    # def CnkD(n,k):
    #     C = defaultdict(int)
    #     for row in range(n+1):
    #         C[row,0] = 1
    #         for col in range(1,k+1):
    #             if col <= row:
    #                 C[row,col] = C[row-1,col-1]+C[row-1,col]
    #     return C[n,k]
    symbols = re.findall("\D+", formula)
    atom_counts = [int(x) for x in re.findall("\d+", formula)]
    if not len(symbols) == len(atom_counts):
        raise ValueError("Invalid formula")
    
    formus = {symbols[i]:atom_counts[i] for i in range(len(symbols))}
    theoretical_isotopic_cluster = isotopic_variants(formus, npeaks=5, charge=charge)
    mz = [peak.mz for peak in theoretical_isotopic_cluster]
    intens = [peak.intensity for peak in theoretical_isotopic_cluster]
    return pd.DataFrame({'mz':mz,'intensity':intens})
 
    

def compare_isotope(measured, expected, tolerance=5e-3):
    '''
    Task: 
        Compare theoretical isotope distribution and measured isotope distribution
    Parameters:
        measured: DataFrame, measured isotope distribution
        expected: DataFrame, theoretical isotope distribution
        tolerance: float, m/z tolerance
    '''
    measured = measured.copy()
    expected = expected.copy()
    if (type(measured) is not pd.DataFrame) or (type(expected) is not pd.DataFrame):
        raise ValueError('input data must be pandas.DataFrame')
    measured['intensity'] = measured['intensity']/max(measured['intensity'])
    expected['intensity'] = expected['intensity']/max(expected['intensity'])
    expected_m0 = expected['intensity'][0]
    measured_m0 = measured['intensity'][0]
    expected_m1 = expected['intensity'][1]
    measured_m1 = measured['intensity'][1]
    expected_m2 = expected['intensity'][2]
    measured_m2 = measured['intensity'][2]
    score = (np.e**(-0.5*((expected_m1 - measured_m1)/tolerance)**2) +
             np.e**(-0.5*((expected_m2 - measured_m2)/tolerance)**2))/2
    return score 

def re_formula(fm):
    """
    分子式修饰，将分子式整理为字母+数字形式。元素有两种表示形式
    1）单个大写字母
    2）大写+小写
    """
    re_sym = []
    sym = re.findall('[A-Z][a-z]*\d*',fm)
    for s in sym:
        if not re.search('\d', s):
            s += '1'
        re_sym.append(s)
    fm = ''.join(re_sym)
    return fm

def formula_jis(fm1,fm2,mode='-'):
    import re
    import numpy as np
    #前提是分子式均为str变量
    if type(fm1) is str and type(fm2) is str:
        fm1 = re_formula(fm1)
        fm2 = re_formula(fm2)
        symbols1 = re.findall("\D+", fm1)
        atom_counts1 = [int(x) for x in re.findall("\d+", fm1)]
        #在分子式中某元素只有1个会出现不匹配问题
        if len(atom_counts1) < len(symbols1):
            atom_counts1.append(1)
        symbols2 = re.findall("\D+", fm2)
        atom_counts2 = [int(x) for x in re.findall("\d+", fm2)]
        if len(atom_counts2) < len(symbols2):
            atom_counts2.append(1)
        # ast1 = [s in symbols1 for s in symbols2]
        #设定好元素的种类及顺序
        all_k = ['C','H','O','N','P','S','Cl','Na','K','+','-']
        # assert set(symbols1).issubset(all_k),'{} Not in set'.format(fm1)
        # assert set(symbols2).issubset(all_k),'{} Not in set'.format(fm2)
        if set(symbols1).issubset(all_k) and set(symbols2).issubset(all_k):
            pass
        else:
            return None
        dic1 = dict()
        dic2 = dict()
        # zip有问题
        for s in all_k:
            if s in symbols1:
                a = atom_counts1[symbols1.index(s)]
                dic1.update({s:a})
            else:
                dic1.update({s:0})
        for s in all_k:
            if s in symbols2:
                a = atom_counts2[symbols2.index(s)]
                dic2.update({s:a})
            else:
                dic2.update({s:0})
        if mode == '-':
            res = np.array(list(dic1.values())) - np.array(list(dic2.values()))
        elif mode == '+':
            res = np.array(list(dic1.values())) + np.array(list(dic2.values()))
        #加减片段分开
        out_pos = []
        out_neg = []
        for s,a in zip(all_k,res):
            if a > 0:
                if a ==1:
                    a =''
                    out_pos.append(s+str(a))
                else:
                    out_pos.append(s+str(a))
            elif a < 0:
                a = abs(a)
                if a == 1:
                    a = ''
                    out_neg.append(s+str(a))
                else:
                    out_neg.append(s+str(a))
        ps = ''.join(out_pos)
        ng = ''.join(out_neg)
        if len(ps) > 0:
            ps = '+'+ps
        if len(ng) > 0:
            ng = '-'+ng
        out_formula = ','.join([ps,ng])
        return out_formula

def get_formula(mode,mass,adduct=None,mDa=5,ppm=10,elements = ['C','H','O','N','P','S']):
    formuladb = pd.read_csv("Source/new_formula_db.csv")
    '''
    Task: 
        Search formula from formula database.
    Parameters:
        mass: float, exact mass of compound
        ppm: float, ppm
    '''
    POS_ADD_TB = pd.DataFrame({"Adduct":['[M+H]+','[M+NH4]+','[M+Na]+'],
                               "Mz":[1.007825,18.034374,22.989770],
                               "Charge":[1,1,1],
                               "Ratio":[1,1,1]})
    NEG_ADD_TB = pd.DataFrame({"Adduct":['[M-H]-','[M+FA-H]-','[M+Cl]-','[2M-H]-'],
                               "Mz":[-1.007825,44.997655,34.968853,-1.007825,],
                               "Charge":[1,1,1,1],
                               "Ratio":[1,1,1,2]})
    res = []
    elements = elements
    def search_formula(mode,mass,formuladb,mDa,ppm,elements = elements):
        mmin1 = mass - mass*ppm/10**6
        mmax1 = mass + mass*ppm/10**6
        mmin2 = mass - mDa*1e-3
        mmax2 = mass + mDa*1e-3
        # if mode == "-":
        #     formulaDB = formulaDB[formulaDB['IonMode']=='Negative']
        # elif mode == '+':
        #     formulaDB = formulaDB[formulaDB['IonMode']=='Positive']
        lf = bisect.bisect_left(formuladb['calc_mz'], min(mmin1,mmin2))
        rg = bisect.bisect_right(formuladb['calc_mz'], max(mmax1,mmax2))
        formulas = list(formuladb['Formula'][lf:rg])
        ms_errors = [(mass-i)*1e3 for i in list(formuladb['calc_mz'][lf:rg])]
        clean_fm = []
        clean_err = []
        ms_er_score = []
        for f,p in zip(formulas,ms_errors):
            # 获取所有元素
            e1 = re.findall('[A-Z][a-z]*',f)
            # 确定e1是要求元素集合的子集
            if len(set(e1).union(set(elements))) == len(elements):
                s = np.e**(-0.5*(p/mDa)**2)
                clean_fm.append(f)
                clean_err.append(p)
                ms_er_score.append(s)
        return clean_fm,clean_err,ms_er_score

    if mode == "+":
        Adducts = POS_ADD_TB.copy()
    elif mode == "-":
        Adducts = NEG_ADD_TB.copy()
    if adduct in list(Adducts.Adduct):
        _add = Adducts[Adducts['Adduct']==adduct]
        exms = float((mass - _add.Mz)/_add.Ratio)
        clean_fm,clean_err,ms_er_score = search_formula(mode,exms,formuladb,mDa,ppm)
        temp = pd.DataFrame({"pepmass":mass,
                                 "Adduct":adduct,
                                 "Formula":clean_fm,
                                 "ppm":clean_err,
                                 "formulascore":ms_er_score})
        # if len(temp)>0:
        return temp
    for a,b,c,d in zip(Adducts.Mz,Adducts.Adduct,Adducts.Charge,Adducts.Ratio):
        exms = float((mass - a)/d)
        clean_fm,clean_err,ms_er_score = search_formula(mode,exms,formuladb,mDa,ppm)
        res.append(pd.DataFrame({"pepmass":mass,
                                 "Adduct":b,
                                 "Formula":clean_fm,
                                 "ppm":clean_err,
                                 "formulascore":ms_er_score}))
    return pd.concat(res).reset_index(drop=True)

def cal_exactmass(formula):
    massmap = {"C":12.000000,
               "H":1.007825,
               "O":15.994915,
               "N":14.003074,
               "P":30.973762,
               "S":31.972071,
               }
    elements = re.findall('([A-Z][a-z]*)', formula)
    x = [e in ['C','H','O','N','P','S'] for e in elements]
    if not False in x:
        formula_p = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        vec1 = np.zeros(len(elements))
        vec2 = np.zeros(len(elements))
        for i in range(len(formula_p)):
            ele = formula_p[i][0]
            num = formula_p[i][1]
            
            if num == '':
                num = 1
            else:
                num = int(num)
            vec1[i] = massmap[ele]
            vec2[i] = num
        return np.dot(vec1,vec2)
    else:
        return  0
    
# formulaDB = pd.read_csv(r'Source/MSFormuladb.csv')
# calc_mz = [cal_exactmass(f) for f in formulaDB['Formula']]
# formulaDB['calc_mz'] = calc_mz
# formulaDB.to_csv(r'Source/MSFormuladb.csv',index=False)

# formulaDB = pd.read_excel(r'Source/MassKGdatabaseS2.xlsx')
# formulaDB = formulaDB['Molecular_Formula'].drop_duplicates()
# calc_mz = []
# for f in formulaDB:
#     try:
#         _ = cal_exactmass(f) 
#         calc_mz.append(_)
#     except Exception as e:
#         calc_mz.append(0)
# formulaDB = pd.concat([formulaDB,pd.Series(calc_mz)],axis=1)
# formulaDB.columns = ['Formula','calc_mz']
# formulaDB.to_csv(r'Source/MSFormuladb_cnt.csv',index=False)


# formulaDB = pd.read_csv(r'Source/MSFormuladb_cnt.csv')
# formulaDB = formulaDB[formulaDB['calc_mz']<=1500].reset_index(drop=True)
# formulaDB.to_csv(r'Source/MSFormuladb_cnt.csv',index=False)
# calc_mz = formulaDB['calc_mz']

# calc_mz = [str(a) for a in calc_mz]
# fst = []
# scd = []
# for a in calc_mz:
#     _ =a.split(".")
#     if len(_)==2:
#         fst.append(eval(_[0][0]))
#         scd.append(eval(_[1][0]))
#     else:
#         fst.append(eval(_[0][0]))
#         scd.append(0)






# from sklearn.cross_decomposition import PLSRegression
# from sklearn.model_selection import cross_validate
# plsr = PLSRegression()
# cross_validate(plsr, X=fst,y=scd)



def get_formula2(mode,rt,mass,ms2,ms1table,formuladb,mDa=5,ppm=10,binsize=100,rt_diff=0.05):
    '''
    Task: 
        Search formula from formula database.
    Parameters:
        mass: float, exact mass of compound
        ppm: float, ppm
    '''
    H_1 = 1.007825
    K_39 = 38.963708
    Na_23 =  22.989770
    NH4 = 18.034374
    Cl_35 = 34.968853
    CHO2 = 44.997655
    H3O = 19.018385
    HO = 17.002735
    C2H3O2 = 59.013305
    res = []
    
    def search_formula(mode,mass,formuladb,mDa,ppm):
        elements = ['C','H','O','N','P','S']
        mmin1 = mass - mass*ppm/10**6
        mmax1 = mass + mass*ppm/10**6
        mmin2 = mass - mDa*1e-3
        mmax2 = mass + mDa*1e-3
        # if mode == "-":
        #     formulaDB = formulaDB[formulaDB['IonMode']=='Negative']
        # elif mode == '+':
        #     formulaDB = formulaDB[formulaDB['IonMode']=='Positive']
        lf = bisect.bisect_left(formuladb['calc_mz'], min(mmin1,mmin2))
        rg = bisect.bisect_right(formuladb['calc_mz'], max(mmax1,mmax2))
        formulas = list(formuladb['Formula'][lf:rg])
        ms_errors = [(mass-i)*1e3 for i in list(formuladb['calc_mz'][lf:rg])]
        clean_fm = []
        clean_err = []
        ms_er_score = []
        for f,p in zip(formulas,ms_errors):
            # 获取所有元素
            e1 = re.findall('[A-Z][a-z]*',f)
            # 确定e1是要求元素集合的子集
            if len(set(e1).union(set(elements))) == len(elements):
                s = np.e**(-0.5*(p/mDa)**2)
                clean_fm.append(f)
                clean_err.append(p)
                ms_er_score.append(s)
        return clean_fm,clean_err,ms_er_score
    
    adduct_table = pd.DataFrame({"adduct":['[M+H]','[M+Na]','[M+NH4]'],
                                "mzdiff":[1.007825,22.9898,18.0344]})
    adduct_table['mzdiff'] = adduct_table['mzdiff'] - 1.007825
    subtable = ms1table[(abs(ms1table['PRECURSORMZ']-mass)<=binsize)&(abs(ms1table['RETENTIONTIME']-rt)<=0.01)]
    subtable = subtable.reset_index(drop=True)
    addcts = []
    diffs = []
    for j in range(subtable.shape[0]):
        for jj in range(adduct_table.shape[0]):
            _ = abs(subtable['PRECURSORMZ'][j] - mass- adduct_table['mzdiff'][jj])
            if _ <=0.01:
                aa = adduct_table['adduct'][jj]
                exms = subtable['PRECURSORMZ'][j] - adduct_table['mzdiff'][jj] - 1.007825
                clean_fm,clean_err,ms_er_score = search_formula(mode,exms,formuladb,mDa,ppm)
                res.append(pd.DataFrame({"pepmass":mass,
                                         "Adduct":aa,
                                         "Formula":clean_fm,
                                         "ppm":clean_err,
                                         "formulascore":ms_er_score}))
                continue
                
    return pd.concat(res).reset_index(drop=True)