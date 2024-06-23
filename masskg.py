from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors
from tqdm import tqdm, trange
from msp_writer import parse_MSDIAL,msptranslator
from mgf_translator import mgftranslator
import numpy as np
import pandas as pd
from utils import *
import os
import argparse
import re
import time
import json
from collections import defaultdict
from _mysql import myclient
import uuid
from flask import Flask, jsonify, render_template, request
import json
import warnings
from flask.json.provider import DefaultJSONProvider, _default as FlaskDefault

        

warnings.filterwarnings("ignore")







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
        # 递归生成字典
        result["children"] = [x.to_dict() for x in self.children]
        return result
    
class TreeProcessor:
    def __init__(self, pair: list, info: dict, d_type="json"):
        """

        :param pair: [(0, 1), (0, 2)...] parent-child pairs
        :param info: {0 : {"name": xxx}} not inform
        :param d_type: return json dict
        """
        self._pair = pair
        self._info = info
        self.d_type = d_type
        self.root_id, self._pair_map, self._ids = self._process_data()

    def _process_data(self):
        """
        trans  [(0, 1), (0, 2)...]  pars into
        {father: [child, child]}
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
        feature_table = feature_table[["SCANNUMBER","RETENTIONTIME","PRECURSORMZ","INTENSITY","PRECURSORTYPE"]]
        feature_table.columns = ['ID','Rt','Mz','Size','Adduct']
        pepmass = list(feature_table['Mz'].astype(float))
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
        raise TypeError("illegal input! file format should be in [txt, msp, mgf, csv, xlsx],your file format is {}".format(appdixs))
    # sort the query table by Size and filter
    feature_table = feature_table.sort_values(by=['Size'],ascending=False).iloc[:500]
    ms2_list = [ms2_list[i] for i in feature_table.index]
    test_ms = [test_ms[i] for i in feature_table.index]
    pepmass = [pepmass[i] for i in feature_table.index]
    feature_table = feature_table.reset_index(drop=True)
    return pepmass,test_ms,feature_table            
def API(project_path,mode,k,thresh,dbparams,developer=False):
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
    datapath = os.path.join(project_path,'MassKG_INPUTS',subpath)
    outputpath = os.path.join(project_path,'MassKG_OUTPUTS')
    fragpath = os.path.join(project_path,'insilicoFRAGs')
    assert os.path.isdir(datapath),'Please select a folder'
    f_names = os.listdir(datapath)
    for f in f_names:        
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
            pepmass = list(feature_table['PEPMASS'].astype(float))
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
                x_0 = x_0[[len(_)>0 for _ in x_1]] # filter candidates without matched fragments
                x_1 = [_ for _ in x_1 if len(_)>0][:k]
                spectra = x_0.iloc[:k,:].copy()
                write_fragments = []
                for n in range(len(x_1)):
                    fragments = ''
                    interactions = x_1[n][1].copy()
                    infos = x_1[n][0].copy()
                    if not len(x_1[n][1]) > 0: # no nl loss mathced
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
                # if not developer:
                #     x_0 = x_0[show_cols]
                # test_ms[i].to_csv(os.path.join(temp_path,'{}-{}_fragments.csv'.format(i,name_ids[i])),index=False)
                summerize_table.append(spectra)
        summerize_table = pd.concat(summerize_table)
        summerize_table.to_csv(os.path.join(temp_path,'Summarysheet.csv'),index=False)
        print(f.split('.')[0],' Finished')
            

# In[]
from flask import *
import json
from werkzeug.utils import secure_filename
    

if __name__ == "__main__":

    # loading datasets
    mstable_p,ms2_list_p = msptranslator('Source/all_pos_with_type.msp')
    ms2_list_p = [pd.DataFrame(m,columns=['mz','intensity']) for m in ms2_list_p]
    mstable_n,ms2_list_n = msptranslator(r'Source/all_neg_with_type.msp')
    ms2_list_n = [pd.DataFrame(m,columns=['mz','intensity']) for m in ms2_list_n]     
    formulaDB = pd.read_csv(r'Source/new_formula_db.csv')
    mstable_p = mstable_p.apply(lambda x:x.str.strip())
    mstable_n = mstable_n.apply(lambda x:x.str.strip())
    ontos_p = [[a.split(":")[1] if len(a.split(":"))>1  else "" for a in m.split(";")] for m in mstable_p['Ontology']]
    ontos_p = pd.DataFrame(ontos_p,columns=['Kingdom', 'Superclass', 'Class', 'Subclass'])
    mstable = pd.concat([mstable_p,ontos_p],axis=1)
    ontos_n = [[a.split(":")[1] if len(a.split(":"))>1  else "" for a in m.split(";")] for m in mstable_n['Ontology']]
    ontos_n = pd.DataFrame(ontos_n,columns=['Kingdom', 'Superclass', 'Class', 'Subclass'])
    mstable_n = pd.concat([mstable_n,ontos_n],axis=1)
    mstable_p = pd.concat([mstable_p,ontos_p],axis=1)
    # fragments = [pd.read_csv() for f in os.listdir()]
    # server buiding 
    UPLOAD_FOLDER = '/home/websites/flask/masskg/uploads' # local path 
    ALLOWED_EXTENSIONS = {'txt', 'csv','msp','mgf'}
    SECRET_KEY = 'herewego'
     
    app = Flask(__name__,
                template_folder=r'./template',
                static_folder=r'./static',)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SECRET_KEY'] = os.urandom(24)
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route("/masskg")
    def index():
        return render_template('index.php')
    @app.route("/masskg/batch")
    def batch():
        return render_template('batch.php')
    @app.route("/masskg/search")
    def search():
        return render_template('search2.php')
    @app.route("/masskg/statistics")
    def statistics():
        return render_template('statistics.php')
    @app.route("/masskg/statistics/res",methods=['GET','POST'])
    def statres():
        conn = pymysql.connect(  # connet mysql
            user="root",
            password="123456",
            host="127.0.0.1",
            database='masskg', 
            charset="utf8"
            )
        cursor = conn.cursor()
        sql = "SELECT Class,COUNT(*) as cnt FROM masskgdatabasev3_2 GROUP BY Class ORDER BY cnt DESC";
        cursor.execute(sql)
        retval = cursor.fetchall()
        retval = [{'name':a[0],'value':a[1]} for a in retval if a[0]!='']
        #print(retval)
        return jsonify(retval)
    @app.route("/masskg/manuals")
    def manual():
        return render_template('manual.php')
    @app.route("/masskg/browser")
    def browser():
        return render_template('browser3.php')
    @app.route("/masskg/network")
    def network():
        return render_template('masskg.html')
    @app.route("/masskg/browser/res",methods=['GET','POST'])
    def browres():
        conn = pymysql.connect(  # mysql
            user="root",
            password="123456",
            host="127.0.0.1",
            database='masskg',
            charset="utf8"
            )
        cursor = conn.cursor()
        cursor.execute(sql)
        retval = cursor.fetchall()
        res = []
        for r in retval:
            res.append({'ick':r[0],
                        'ID':r[13],
                        'smiles':r[1],
                        'mf':r[4],
                        'ms':r[6],
                        'subclass':r[11],
                        'outlink':r[12]})
        res = {'data':res}
        return jsonify(res)
    @app.route("/masskg/test/<filename><_link>",methods=['GET','POST'])
    def test(filename,_link):
        return render_template('test.php',filename='a')
    @app.route("/masskg/upload_file", methods=['GET', 'POST'])
    def upload_file():
        uid = uuid.uuid4()
        resp = make_response("upload success")
        resp.set_cookie("filename", "NGE-1")
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                print(file.filename)
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                savepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uid))
                if not os.path.exists(savepath):
                    os.makedirs(savepath, exist_ok=True)
                file.save(os.path.join(savepath, filename))
                print(uid)
                resp.set_cookie("uid", str(uid))
                resp.set_cookie("filename", filename)
                resp.set_cookie("savepath", savepath)
                session['filename'] = filename
                session['savepath'] = savepath
                #return redirect(url_for('batch'))
                #return render_template('batch.php', filename=filename)
                return jsonify([filename])
        return redirect(url_for('batch'))

    @app.route("/masskg/Oneres",methods=['GET','POST'])
    def Oneres():
        t0 = time.time()
        project_path = '/home/websites/flask/masskg'
        mDa = 5
        k = 5
        data = json.loads(request.form.get('data'))
        data = {d['name']:d['value'] for d in data}
        mz = eval(data['miw'])
        mode = data['mode']
        ms2 = data['ms2']
        ppm = eval(data['ppm'])
        formula = data['formula']
        database = 'masskgdatabasev3_2'
        if mode == "-1":
            dbparams = [f'{database}',mstable_p,ms2_list_p,formulaDB]
            subpath = 'POS'
            mode = "+"
        elif mode == '1':
            dbparams = [f'{database}',mstable_n,ms2_list_n,formulaDB]
            subpath = 'NEG'
            mode = "-"
        formuladb = dbparams[3]
        ms2 = ms2.split("\r\n")
        ms2 = [m.strip() for m in ms2]
        ms2 = np.array([m.split(" ")if" " in m else m.split("\t") for m in ms2],dtype=float)
        
        fragpath = os.path.join(project_path,'insilicoFRAGs')
        test_ms = prems2(ms2)
        pepmass = [mz]
        if formula == '':
          res = [get_formula(mode,m,formuladb,mDa=mDa,ppm=ppm) for m in pepmass]
          x = get_rank(res[0],test_ms,mode,fragpath,dbparams)
        else:
          query = pd.DataFrame({'pepmass':pepmass,"Formula":[formula],"Adduct":[""],"formulascore":[1],"ppm":[ppm]})
          x = get_rank(query,test_ms,mode,fragpath,dbparams)
        ########################################

        ########################################
        if len(x) > 0:
            x_0 = x[0]
            # x_0['Mz'] = mz
            x_1 = x[1]# matched fragments
            x_0 = x_0[[len(x[0])>0 for x in x_1]] 
            x_1 = [x for x in x_1 if len(x[0])>0][:k]
            # nloss = [assign_nl(x, mode) for x in x_1]

            # nloss = [assign_nl2(x, mode) for x in x_1]
            nloss = x_1.copy()
            for n in range(len(nloss)):
                if not len(nloss[n][1]) > 0: 
                    continue
                infos = nloss[n][0][['ID','mz']]
                root_id = max(infos['ID']) # set up root_id
                infos = {str(i):{"name":str(j)}for i,j in zip(list(infos.ID),list(infos.mz))}
                relation = [(str(i),str(j)) for i,j in zip(nloss[n][1]['Source_ID'],nloss[n][1]['Target_ID'])]
                add_relation = [(str(root_id),str(j)) for j in nloss[n][1]['Source_ID'] if j!= root_id]
                relation = list(set(relation + add_relation))
                tp = TreeProcessor(relation, infos)
                tree_dict = tp.tree
                # with open(os.path.join(outputpath,'nltree_{}.json').format(n),'w') as f:
                #     f.write(tree_dict)
            

            for _ in x_0:
                if "score" in _:
                    x_0[_] = x_0[_].astype(float).round(2)
            scores = []
     
            for i in range(x_0.shape[0]):
                scores.append("formula score : {}\r\nstructure score: {}\r\nfrag me score: {}\r\nneutral loss score: {}\r\nfinal score: {}".format(
                    x_0['formulascore'][i],x_0['structure_score'][i],x_0['fragmes_mse'][i].round(2),x_0['nlscore'][i],x_0['final_score'][i]))
                
            x_0 = x_0.drop(columns=['ppm','pepmass','punish_score','fragmes_mse',
                                    'formulascore','structure_score','nlscore','final_score'])
            x_0 = pd.concat([x_0,pd.Series(scores,name='Score')],axis=1)
            x_0.insert(0,'Rank',range(1,len(x_0)+1))
            # numpy to list→json
            x_0 = list(json.loads(x_0.T.to_json()).values())
            t1 = time.time()
            print("finished!!!")
            print('Time consumption {:.2f} s'.format((t1-t0)))
            return jsonify([{'tree':tree_dict},{'data':x_0}])
      # no candidates
        else:
            return jsonify([{'tree':None},{'data':None}])
    
    @app.route("/masskg/batchres",methods=['GET','POST'])
    def batchres():
        print("loaded")
        t0 = time.time()
        project_path = '/home/websites/flask/masskg'
        mDa = 5

        data = json.loads(request.form.get('data'))
        data = {d['name']:d['value'] for d in data}
        k = eval(data['topk'])
        mode = data['mode']
        ppm = eval(data['ppm'])
        thresh = eval(data['threshold'])
        database = 'masskgdatabasev3_2'
        if mode == "-1":
            dbparams = [f'{database}',mstable_p,ms2_list_p,formulaDB]
            subpath = 'POS'
            mode = "+"
        elif mode == '1':
            dbparams = [f'{database}',mstable_n,ms2_list_n,formulaDB]
            subpath = 'NEG'
            mode = "-"
        formuladb = dbparams[3]
        # filepath = request.cookies.get("savepath")
        filepath = session.get("savepath")
        datapath = UPLOAD_FOLDER
        outputpath = os.path.join(project_path,'MassKG_OUTPUTS')
        fragpath = os.path.join(project_path,'insilicoFRAGs')
        # filename = request.cookies.get('filename') # read file name
        filename = session.get('filename')
        if filename == None:
            filename = "NEG-1.txt"
            
        pepmass,test_ms,feature_table = read_file(filepath, filename, thresh)
        print(filename.split('.')[0],' Start Analyzing... ...' )
        if not "ID" in feature_table:
            feature_table['ID'] = range(feature_table.shape[0])
        name_ids = feature_table['ID']

        if "Adduct" in feature_table:
            adducts = list(feature_table['Adduct'])
            feature_table.rename(columns={"Adduct":"PRECURSORTYPE"},inplace=True)
            res = [get_formula(mode,m,formuladb,adduct=a,mDa=mDa,ppm=ppm) for m,a in zip(pepmass,adducts)]
        else:
            res = [get_formula(mode,m,formuladb,adduct=None,mDa=mDa,ppm=ppm) for m in pepmass]
            
            # res is a list of df, columns=[pepmass, Adduct, Formula, ppm, formulascore]
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
        summerize_table.reset_index(inplace=True,drop=True)
        summerize_table = summerize_table[['ID_Rank', 'ID', 'Rt', 'Mz', 'Size','Adduct',
       'Formula', 'ppm', 'formulascore', 'Exact_mass', 'NAME', 'InChIKey',
       'SMILES', 'MassKGID', 'Kingdom', 'Superclass', 'Class', 'Subclass',
       'punish_score', 'structure_score', 'fragmes_mse', 'level', 'nlscore',
       'final_score', 'fragments']]
        result_path = os.path.join(filepath,'Summarysheet.csv')
        summerize_table.to_csv(result_path,index=False)
        summerize_table = list(json.loads(summerize_table.T.to_json()).values())

        return jsonify({'data':summerize_table})
