import re
import os
import numpy as np
import pandas as pd

def prems2(_ms2,max=100,precision=2,profile=False):
    # profile to centriod
    binsize = 0.5
    _ms2 = pd.DataFrame(_ms2)
    _ms2.columns = ['mz','intensity']
    _ms2 = _ms2.sort_values(by=['intensity'],ascending=False).reset_index(drop=True)

    if profile:
        res = []
        for i in range(_ms2.shape[0]):
            mz = list(_ms2['mz'])[i]
            temp = _ms2[(_ms2['mz']>mz-binsize)&(_ms2['mz']<mz+binsize)].iloc[0]
            res.append(temp)
        res = pd.DataFrame(res)
        res = res.drop_duplicates().sort_values(by=['mz']).reset_index(drop=True)
    else:
        res = _ms2.copy()
    res['intensity'] = res['intensity'].round(precision)
    res['intensity'] = res['intensity']/res['intensity'].max()*max
    res = res.reset_index()
    res.columns = ['ID','mz','intensity']
    return res

def sum_ms2(ms2_table,upper=100):
    """
    input a ms2 table consist of 2 columns: 0 is the mz value and 1 is intensity
    """
    ms2_table = ms2_table.sort_values(by=0).reset_index(drop=True)
    mzs = ms2_table[0]
    intensity = ms2_table[1]
    new_mz = []
    new_int = []
    if len(ms2_table) > 1:
        for i in range(len(ms2_table)-1):
            m = [mzs[i]]
            inten = [intensity[i]]
            if abs(mzs[i]-mzs[i+1]) < 5e-3 or abs(mzs[i]-mzs[i+1])/mzs[i] < 20e-6:
                m.append(mzs[i+1])
                inten.append(intensity[i+1])
            else:
                new_mz.append(np.mean(m))
                new_int.append(np.sum(inten))
        if i == len(ms2_table) - 2:
            if abs(mzs[i]-mzs[i+1]) < 5e-3 or abs(mzs[i]-mzs[i+1])/mzs[i] < 20e-6:
                new_mz.append(np.mean(m))
                new_int.append(np.sum(inten))
            else:
                new_mz.append(mzs[i+1])
                new_int.append(intensity[i+1])
        res = pd.DataFrame([new_mz,new_int]).T
        res[1] = res[1]/max(res[1])*upper
        res[1] = res[1].astype(int)
    else:
        res = ms2_table
    return res
            

def mgftranslator(path_mgf,maxlen=0):

    MS2 = []
    MSlist = []
    features = []
    VALUES = []
    MSlist2 = []
    count = 0
    with open(path_mgf) as f:
        raw_data=f.readlines()# read mgf file to a list line by line
    if re.match('BEGIN IONS',raw_data[0]): 
        for j in range(1,len(raw_data)):
            if re.match('^\d',raw_data[j]):
                break
            else:
                features.append(raw_data[j])
    features = [s.split('=',1) for s in features]
    KEYS = [f[0] for f in features]

    lenz = len(KEYS)

  
    
    for i in range(len(raw_data)):
        if re.match('BEGIN IONS',raw_data[i]):
            values = []
            for j in range(lenz):
                v = raw_data[i+j+1].split('=',1)[1] 
                v = v.replace(r'\n','')
                v = v.replace(r'\t',' ')
                v = v.strip()
                values.append(v)
            VALUES.append(values)
        elif re.match('^\d',raw_data[i]):
            MS2.append(raw_data[i])
        elif re.match('END IONS',raw_data[i]):
            m=pd.DataFrame(MS2)
            try:
                m = m[0].str.replace(r'\n','')
                m = m.str.replace(r'\t',' ')
                m = m.str.strip()
                m = m.str.split(' ',expand=True)
                MSlist.append(m.apply(pd.to_numeric))
            except  KeyError:
                MSlist.append(pd.DataFrame([[0,0]]))
                print(i,"has no fragments")
            count += 1
            MS2 = []
    print('total ============= {} blocks finished'.format(count))
    feature_table = pd.DataFrame(VALUES,columns=KEYS) 

    lenth = maxlen
    MSlist = [sum_ms2(m) for m in MSlist if len(MSlist)>1]
    if maxlen == 0:
        MSlist = [m.values for m in MSlist]
        return feature_table,MSlist
    else:
        for d in MSlist:
            l=[]
            ind=0
            m=d.sort_values(by=1,ascending=False)
            m=m.reset_index(drop=True)
            while ind < lenth:
                if len(m)<lenth:
                    if ind<len(m):
                        l.append(list(m.iloc[ind]))
                    else:               
                        l.append([0,0])
                else:
                    l.append(list(m.iloc[ind]))
                ind+=1
            MSlist2.append(l)
        return feature_table,MSlist2   
    
def get_ms(ms_,maxlen=20):
    ms_list = []
    for mm in ms_:
        if type(mm) is str:
            if ":" in mm:
                mmm = mm.split(" ")
                m = [x.split(":") for x in mmm]
            else:
                mmm = mm.split(";")
                m = [x.split(" ") for x in mmm]
            for _ in m:
                if len(_) != 2:
                    print(_) 
            temp = pd.DataFrame(m)
            temp.columns = ['mz','intensity']
            temp = temp[temp['intensity']!='']  
            temp = temp.dropna().astype(float)
            temp = temp.sort_values(by='intensity',
                                    ascending=False).reset_index(drop=True)
            if maxlen:
                temp = temp.iloc[:maxlen,:]
            ms_list.append(temp)
        else:
            ms_list.append(pd.DataFrame(columns=['mz','intensity'])) 
    return ms_list
    
columns = ['NAME','PRECURSORMZ','PRECURSORTYPE','FORMULA',
       'Ontology','INCHIKEY','SMILES','RETENTIONTIME','CCS',
       'IONMODE','INSTRUMENTTYPE','INSTRUMENT','COLLISIONENERGY',
       'Comment','Num Peaks']
def parse_MSDIAL(df,iso=False,maxlen=10,tolerance=5e-3,keepms2=False):
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
            # print(i)
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
    res = pd.DataFrame(res)
    print('total ============= {} blocks finished'.format(len(res)))
    # res = res[columns]
    return res,MSlist2



def Ontology(smiles):
    from pyclassyfire import client
    qid = client.structure_query(smiles,'smiles_test')
    res = client.get_results(qid,'csv')
    k = re.findall(r'Kingdom: [A-Za-z| ]+', res)
    k = k[0] if len(k)>0 else ''
    sc = re.findall(r'Superclass: [A-Za-z| ]+', res)
    sc = sc[0] if len(sc)>0 else ''
    c = re.findall(r'Class: [A-Za-z| ]+', res)
    c = c[0] if len(c)>0 else ''
    subc = re.findall(r'Subclass: [A-Za-z| ]+', res)
    subc = subc[0] if len(subc)>0 else ''
    return ';'.join([k,sc,c,subc])


def NPOntology(s):
    from urllib import request
    import json
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    url = 'https://npclassifier.ucsd.edu/classify?smiles={}'
    headers=('User-Agent',
        'Mozilla/5.0(Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36')
    opener = request.build_opener()
    opener.addheaders=[headers]
    try:
        data = opener.open(url.format(s),timeout=1).read()
        data = json.loads(data.decode())
        o = data['pathway_results'][0]
        return o
    except Exception as e:
        print('ERROR',e)
        return 'No Ontology'
    
        
def cal_exactmass(formula):
    massmap = {"C":12.000000,
               "H":1.007825,
               "O":15.994915,
               "N":14.003074,
               }
    elements = re.findall('([A-Z][a-z]*)', formula)
    x = [e in ['C','H','O','N'] for e in elements]
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
def reonto_library(msp_path,outpath):
    feature_table,ref_ms2 = msptranslator(msp_path)
    ref_ms2 = [pd.DataFrame(r) for r in ref_ms2]
    Ontos = []
    for i in range(len(feature_table['SMILES'])):
        smiles = feature_table.iloc[i]['SMILES']
        try:
            o = Ontology(smiles)
        except Exception as e:
            print(e)
            o = ';'
        Ontos.append(o)
        print(i)
    
    feature_table['Ontology'] = Ontos
    s = writemsp(feature_table,ref_ms2,outpath)


def read_file(datapath,f,thresh):
    appdix = f.split('.')[-1]
    adducts = None
    if  appdix == 'txt':
        df = pd.read_table(os.path.join(datapath,f))
        if "Size" in df:    
            df = df[df['Size']>thresh].reset_index(drop=True)
        elif "Area" in df:
            df = df[df['Area']>thresh].reset_index(drop=True)
        iso_list,ms2_list,feature_table = parse_MSDIAL(df,maxlen=50,tolerance=0.001)
        feature_table = feature_table[[len(m)>0 for m in ms2_list]].reset_index(drop=True)
        ms2_list = [m for m in ms2_list if len(m)>0]
        test_ms = [prems2(m) for m in ms2_list] 
        pepmass = list(feature_table['Mz'].astype(float))

    elif appdix == 'msp':
        feature_table,ms2_list = msptranslator(os.path.join(datapath,f),)
        feature_table = feature_table[[len(m)>0 for m in ms2_list]].reset_index(drop=True)
        ms2_list = [m for m in ms2_list if len(m)>0]
        test_ms = [prems2(m) for m in ms2_list] 
        pepmass = list(feature_table['PRECURSORMZ'].astype(float))
    elif appdix == 'mgf':
        feature_table,ms2_list = mgftranslator(os.path.join(datapath,f))
        feature_table = feature_table[[len(m)>0 for m in ms2_list]].reset_index(drop=True)
        ms2_list = [m for m in ms2_list if len(m)>0]
        test_ms = [prems2(m) for m in ms2_list] 
        pepmass = list(feature_table['PEPMASS'].astype(float))
    elif appdix == 'csv':
        df = pd.read_csv(os.path.join(datapath,f))
        if "Size" in df:    
            df = df[df['Size']>thresh].reset_index(drop=True)
        elif "Area" in df:
            df = df[df['Area']>thresh].reset_index(drop=True)
        iso_list,ms2_list,feature_table = parse_MSDIAL(df,maxlen=50,tolerance=0.001)
        feature_table = feature_table[[len(m)>0 for m in ms2_list]].reset_index(drop=True)
        ms2_list = [m for m in ms2_list if len(m)>0]
        test_ms = [prems2(m) for m in ms2_list] 
        pepmass = list(feature_table['Mz'].astype(float))
    elif appdix == 'xlsx':
        df = pd.read_excel(os.path.join(datapath,f))
        if "Size" in df:    
            df = df[df['Size']>thresh].reset_index(drop=True)
        elif "Area" in df:
            df = df[df['Area']>thresh].reset_index(drop=True)
        iso_list,ms2_list,feature_table = parse_MSDIAL(df,maxlen=50,tolerance=0.001)
        feature_table = feature_table[[len(m)>0 for m in ms2_list]].reset_index(drop=True)
        ms2_list = [m for m in ms2_list if len(m)>0]
        test_ms = [prems2(m) for m in ms2_list] 
        pepmass = list(feature_table['Mz'].astype(float))
    else:
        raise TypeError("illegal input! file format should be in [txt, msp, mgf, csv, xlsx]")
    return pepmass,test_ms,feature_table   
