from dis import disco
from os import remove
import os
#from tkinter.tix import InputOnly
from unicodedata import normalize
from xmlrpc.client import Boolean
from rdkit.Chem import PandasTools
from random import randint
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys,Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import inchi as rd_inchi
from json import JSONEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')
from rdkit.Chem.MolStandardize import rdMolStandardize
import json
import re
import math
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode
from st_aggrid.shared import JsCode
import base64
from pkgutil import iter_modules
from io import BytesIO
from PIL import Image

st.session_state["has_run"] = None

def persist_dataframe(df,updated_df_key,col_to_delete):
            # drop column from dataframe
            delete_col = st.session_state[col_to_delete]

            if delete_col in st.session_state[updated_df_key]:
                st.session_state[updated_df_key] = st.session_state[updated_df_key].drop(columns=[delete_col])
            else:
                st.sidebar.warning("Column previously deleted. Select another column.")
            with st.container():
                st.header("**Updated input data**") 
                AgGrid(st.session_state[updated_df_key])
                st.header('**Original input data**')
                AgGrid(df)

def check_if_name_in_column(df: pd.DataFrame, name):
        if name in df.columns:
            pass
        else:
            st.sidebar.error('Please enter a valid column name')
            st.stop()
            #return False
class Commons:
    def __init__(self) -> None:
        pass
    UPDATE_DATAFRAME_KEY = lambda _,string = "": f"update_dataframe-{string}" if string != "" else "update_dataframe"
    DELETE_COL_KEY = lambda _,string = "": f"delete_col-{string}" if string != "" else "delete_col"
    CURATED_DF_KEY = 'Curated'
class Custom_Components:
    commons = Commons()
    update_df = commons.UPDATE_DATAFRAME_KEY
    delete_col_key = commons.DELETE_COL_KEY

    def __init__(self):
        pass
    
    def AgGrid(self,df,key = None,Table_title="Input data"):
        gd = GridOptionsBuilder.from_dataframe(df)
        gd.configure_pagination(enabled=True)
        gd.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        gd.configure_selection('multiple', use_checkbox=False)
        gd = gd.build()
        st.header(f"**{Table_title}**")
        AgGrid(df,key = key,height=500,width=800,gridOptions=gd,columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
    
    def filedownload(self,df,data):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            st.header(f"**Download {data} data**")
            href = f'<a href="data:file/csv;base64,{b64}" download="{data}_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def min_n_max_number_input(self,label,step = 1,min_value = 0,max_value = 1000,key = [None,None],context : st = st.sidebar):
        c = context
        MIN = c.number_input(f"Minimum {label}",step = step,min_value = min_value,max_value = max_value, key = key[0])
        if key[1] in st.session_state:
            if st.session_state[key[1]] <= st.session_state[key[0]]:
                st.session_state[key[1]] = st.session_state[key[0]]+1
            else:
                pass
            MAX = c.number_input(f"Maximum {label}",step = step,min_value = st.session_state[key[1]],max_value = max_value, key = key[1])
            return MIN,MAX
        else:
            MAX = c.number_input(f"Maximum {label}",step = step,min_value = st.session_state[key[0]]+1,max_value = max_value, key = key[1])
            return MIN,MAX

    # Add the upload_file method to the Custom_Components class
    def upload_file(self, custom_title="Upload file", context: st = None, file_type="csv", key="")-> pd.DataFrame:
        if context is None:
            st.subheader(f"{custom_title}")
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="uploader" + key)

            st.markdown("""
            [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/example_modeling_dataset_for_curation.csv)
            """)
        else:
            context.subheader(f"{custom_title}")
            uploaded_file = context.file_uploader("Choose a file", type=["csv", "xlsx"], key="uploader" + "_" + key)

            context.markdown("""
            [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/example_modeling_dataset_for_curation.csv)
            """)

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if not context:
                    if not key:
                        self.AgGrid(df, key="input" + key)
                    else:
                        self.AgGrid(df, key="input" + "_" + key)
                return df
            except Exception as e:
                st.error("Error reading file. Please try again.")
                st.error(e)
        elif not context and uploaded_file is None:
            st.info("Please upload a file.")
        elif context and uploaded_file is None:
            context.info("Please upload a file.")


    def img_tag_generator(self, imgs: list[Image.Image]):
        # Cria o diretório "imgs" se ele não existir
        if not os.path.exists("./imgs"):
            os.makedirs("./imgs")

        img_tag = []
        for i, img in enumerate(imgs):
            img.save(f'./imgs/temp_{i}.png', format='PNG')
            img_str = base64.b64encode(open(f"./imgs/temp_{i}.png", "rb").read()).decode('utf-8')
            img_tag.append(f"data:image/png;base64,{img_str}")

        return img_tag

    def img_AgGrid(self,df,mol_col,title = "Input",key = None):
        ShowImage = JsCode(
            """function (params) {
                    var element = document.createElement("span");
                    var imageElement = document.createElement("img");
                    
                    //document.querySelectorAll('[role="row"]').forEach(function (x){
                    //x.style["height"] = "150px";
                    //});
                    //document.querySelectorAll('[role="gridcell"]').forEach(function (x){
                    //x.style["height"] = "150";
                    //});

                    if (params.data.MOLECULE != '') {
                        imageElement.src = params.data.MOLECULE;
                        
                        imageElement.width = 100;
                        imageElement.height = 100;
                    } 
                    else { imageElement.src = ""; }
                    element.appendChild(imageElement);
                    /*element.appendChild(document.createTextNode(params.value));*/
                    return element;
                    }"""
            )
        mols = [Chem.MolFromSmiles(x) for x in df[mol_col]]
        imgs = [Draw.MolToImage(mol,size=(100,100)) for mol in mols]
        df["MOLECULE"] = self.img_tag_generator(imgs)
        gd = GridOptionsBuilder.from_dataframe(df)
        gd.configure_column("MOLECULE", cellRenderer = ShowImage)
        gd.configure_pagination(enabled = True,paginationAutoPageSize = False, paginationPageSize = 10)
        gd.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
        gd.configure_selection('multiple', use_checkbox=True)
        gd.configure_grid_options(rowHeight=100)
        gd.configure_column("MOLECULE", cellRenderer = ShowImage)
        
        gd = gd.build()
        st.header(f"**{title}**")
        AgGrid(df, width = 800,height=500 ,gridOptions = gd, key=f"IMG_{key}" ,allow_unsafe_jscode = True, columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)

    def persist_df_through_deletion(self,updated_df_key, col_to_delete_key) -> pd.DataFrame:
            # drop column from dataframe
            s_state = st.session_state
            delete_col = s_state[col_to_delete_key]
            if delete_col in s_state[updated_df_key].columns:
                deleted = s_state[updated_df_key].drop(columns=[delete_col])
                #if "input" in s_state:
                    #s_state.pop("input")
                    #st.header("**Updated input data**") 
                #st.info("Deleted col "+delete_col)
                self.AgGrid(df=deleted,Table_title=f"Input without deleted column: {delete_col}")
                return deleted

            else:
                st.warning("Column previously deleted. Select another column.")
    
    def delete_column(self,df,key = "") -> pd.DataFrame:
        
        if df is not None:
            key = self.update_df(key)
            st.session_state[key] = df
            st.write("Delete columns unrelated to descriptors outcome values")
            with st.form(key = "Form-"+self.delete_col_key(key),clear_on_submit=False):
                index = st.session_state[key].columns.tolist().index(
                    st.session_state[key].columns.tolist()[0]
                )
                st.selectbox(
                    "Select columns to delete", options=st.session_state[key].columns, index=index, key=self.delete_col_key(key)
                )
                col1, col_f ,col2 = st.columns([2,2,0.92],gap="large")
                with col1:
                    delete = st.form_submit_button(label="Delete")
                with col_f:
                    undo = st.form_submit_button(label="Undo")
                with col2:
                    proceed = st.form_submit_button("Proceed")
            if delete:
                deleted = self.persist_df_through_deletion(key,self.delete_col_key(key))
                st.session_state["has_run"] = deleted
                return deleted
            if undo:
                st.session_state[key] = df
                st.session_state["has_run"] = None
                self.AgGrid(df,Table_title="Input")
                return df
            if proceed and st.session_state["has_run"] is not None:
                return st.session_state["has_run"]
            if proceed and st.session_state["has_run"] is None:
                return st.session_state[key]
                
    def ReadPictureFiles(self,wch_fl) -> base64:
        try:
            return base64.b64encode(open(wch_fl, 'rb').read()).decode()
        except:
            return ""

    def mol_to_img(self,m):
        dm = Draw.PrepareMolForDrawing(m)
        d2d = Draw.MolDraw2DCairo(250,200)
        dopts = d2d.drawOptions()
        dopts.dummiesAreAttachments=True
        d2d.DrawMolecule(dm)
        d2d.FinishDrawing()
        png_data = d2d.GetDrawingText() 
        png_data = base64.encodebytes(png_data)
        html = str(png_data.decode())
        #html ='data:image/png;base64,%s>'%png_data.decode()
        return html

    def render_image(self,df,img_col):
        ShowImage = JsCode(
            """function (params) {
                    var element = document.createElement("span");
                    var imageElement = document.createElement("img");
                
                    if (params.data.ImgPath != '') {
                        imageElement.src = params.data.ImgPath;
                        imageElement.width="20";
                    } 
                    else { imageElement.src = ""; }
                    element.appendChild(imageElement);
                    element.appendChild(document.createTextNode(params.value));
                    return element;
                    }"""
            )
        
        if df.shape[0] > 0:
            for row in df[img_col]:
                imgExtn = row[-4:]
                row = f'data:image/{imgExtn};base64,' + self.ReadPictureFiles(row)

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column(img_col, cellRenderer = ShowImage)
        vgo = gb.build()
        return AgGrid(df, gridOptions = vgo, height = 150, allow_unsafe_jscode = True )
class Continuous_Duplicate_Remover:
    def __init__(self,df: pd.DataFrame ,duplicate_col: str,value_col: str,convert_to_p: bool,convert_to_mol: bool) :
        self.df = df
        self.duplicate_col = duplicate_col
        self.value_col = value_col
        self.convert_to_p = convert_to_p
        self.convert_to_mol = convert_to_mol
    
    def get_mass(self,smiles):

        parts = re.findall("[A-Z][a-z]?|[0-9]+", smiles)
        mass = 0

        for index in range(len(parts)):
            if parts[index].isnumeric():
                continue

            atom = Chem.Atom(parts[index])
            multiplier = int(parts[index + 1]) if len(parts) > index + 1 and parts[index + 1].isnumeric() else 1
            mass +=  atom.GetMass() * multiplier

        return mass
    
    def isInt(self,i):
        """Check if it is an integer"""
        try:
            int(i)
            return True
        except ValueError:
            return False

    def isFloat(self,f):
        try:
            float(f)
            return True
        except ValueError:
            return False

    # Method that calculates the standard deviation with a number list and the mean, but use np.std() instead
    def stdCalculation(self,numList, mean):
        n = len(numList)+1
        soma = 0
        for x in numList:
            soma += (x-mean)**2
        std = math.sqrt((soma)/n)
        return std

    # Method to calculate the z score for each number in a array of numbers, returns a list of z-scores of each number relevant to the array
    def z_scorer(self,nums: list, mean: float or int, std: float or int):

        z_scores = []
        for x in nums:
            z = (x-mean)/std
            z_scores.append(z)
        return z_scores

    # Most of the action happens here
    def remove_duplicates(self, max_z: float = 2.0, convert_to_p: bool = True) -> pd.DataFrame:
        df = self.df
        name_col = self.duplicate_col
        value_col = self.value_col
        dict_rows = {}
        convert_to_p = self.convert_to_p
        means = {}
        # Comment if you have empty values
        df = df.dropna(axis = 0, how = "all")
        values = [float(num) for num in df.loc[:, value_col]]
        
        names = [str(drugs) for drugs in df.loc[:, name_col]]# if type(drugs) is str]
        mols = [Chem.MolFromSmiles(smi) for smi in names]
        formulas = [CalcMolFormula(mol) for mol in mols if mol is not None]
        for i, name in enumerate(names):
            if name=="" or name=="-":
                name = "Empty"
                
            if name not in dict_rows:
                dict_rows[name] = []
                
                dict_rows[name].append(values[i])
            else:
                dict_rows[name].append(values[i])

        for i,n in enumerate(dict_rows):
            media = []
            if self.convert_to_mol:
                molw = self.get_mass(formulas[i])
            else:
                molw = 1
            if len(dict_rows[n]) > 1 and type(dict_rows[n]) is float:
                means[n] = np.mean(dict_rows[n])

                if dict_rows[n][0] != 0 and means[n]/dict_rows[n][0] !=  1:
                    std = np.std(dict_rows[n])
                else:
                    std = 1

                z = self.z_scorer(dict_rows[n], means[n], std)

                for j, z_value in enumerate(z):
                    if abs(z_value) <= max_z:
                        media.append(dict_rows[n][j])

                med = np.mean(media)
                
                # atrubutes where in the column specified is equal to the name of the current row, and replaces the value (dose) of that row
                if convert_to_p:
                    df.loc[df[name_col] == n, value_col] = -math.log10(med/molw)
                else:
                    df.loc[df[name_col] == n, value_col] = med/molw
            else:
                val = dict_rows[n][0]
                #print(dict_rows[n])
                if convert_to_p:
                    if dict_rows[n][0] <=  0:
                        df.loc[df[name_col] == n, value_col] = val
                    else: 
                        df.loc[df[name_col] == n, value_col] = -math.log10(val/molw)
                else:
                    df.loc[df[name_col] == n, value_col] = val/molw
                    
        dups = len(values) - len(dict_rows.keys())
        #tl = len(dict_rows.keys())
        return df,dups #,tl
class Classification_Duplicate_Remover:

    duplicate_stereo = "InChIKey"
    endpoint_start = 2
    endpoint_end = 13
    fontes = ["fonte","source"]
    pos = ["positive","positivo","mutagenic","yes","1",1,"active"]
    neg = ["negative","negativo","non-mutagenic","non mutagenic","no","0","not mutagenic","not-mutagenic",0,"inactive"]
    non_valids = [np.nan,"",None]
    ames_sorted = True

    def __init__(self, df : pd.DataFrame = None, 
                    duplicate_col = duplicate_stereo, value_col = "Outcome",
                    positive_list = pos, negative_list = neg, 
                    not_valid_list = non_valids, worst_case_scenario = False):    
        self.df = df
        self.duplicate_stereo = duplicate_col
        self.pos = positive_list
        self.neg = negative_list
        #self.fontes = source_col_list
        self.non_valids = not_valid_list
        self.worst_case_scenario = worst_case_scenario
        self.value_col = value_col

    def custom_all(self, iterable, expected_comparison_list):
        for element in iterable:
            if element not in expected_comparison_list:
                return False
        return True
    
    def custom_any(self, iterable, expected_comparison_list):
        for element in iterable:
            if element in expected_comparison_list:
                return True
        return False

    bigger = lambda self,x,y : x if x > y else y
    #smaller = lambda x,y: x if x < y else y
    
    def binary_scorer(self, iterable, positive, negative):
        discordance=0
        real_size=0
        negatives = 0
        positives = 0
        for element in iterable:
            if element in negative:
                negatives+=1
                real_size+=1
            elif element in positive:
                positives+=1
                real_size+=1 
        if real_size > 0:
            discordance = self.bigger(positives, negatives) / real_size
            if discordance == 1:
                discordance = 0 
            return discordance,real_size,positives,negatives
        else: 
            return 0,0 

    def conditions(self, iterable, non_valid_case, neg : list, pos : list):
            if self.custom_all(iterable,neg):
                element = neg[0]
                return element
            elif self.custom_all(iterable,self.non_valids):
                element = non_valid_case
                return element
            elif self.custom_any(iterable,pos):
                element = pos[0]
                return element
            elif not self.custom_any(iterable,pos) and self.custom_any(iterable,neg):
                element = neg[0]
                return element
                
    def remove_duplicates(self) -> pd.DataFrame:
        df = self.df
        name_col = self.duplicate_stereo
        value_cols = self.value_col
        dict_rows = {}
        dups = 0
        pos = self.pos
        neg = self.neg

        values = [val for val in df[value_cols].values]
        names = [drugs for drugs in df.loc[:, name_col] if type(drugs) is str]
        #print(columns)
        for i, name in enumerate(names):
            if name == "" or name == "-":
                name ="Empty"
                
            if name not in dict_rows:
                dict_rows[name] = []
                dict_rows[name].append(values[i])
            else:
                dict_rows[name].append(values[i])
        #print(df.iloc[1,-1])
        for key,value in dict_rows.items():
            #separate cepas from ames            
            #row = df.loc[df[name_col] == key].values.flatten().tolist()
            if len(value) > 1:
                discordance,_,positive_num,negative_num = self.binary_scorer(value,pos,neg)
                if discordance < 0.3:
                    if self.worst_case_scenario:
                        if positive_num > 0:
                            dict_rows[key] = 1
                        else:
                            dict_rows[key] = 0
                    else:
                        if self.bigger(positive_num,negative_num) == positive_num:
                            dict_rows[key] = 1
                        else:
                            dict_rows[key] = 0
            
            df.loc[df[name_col] == key, value_cols] = dict_rows[key]

        dups = len(names) - len(dict_rows.keys())
        return df,dups
class Curation:
    
    def __init__(self,smiles):
        self.curated_smiles = f'curated_{smiles}'
        self.smiles = smiles
    
    def is_smiles_passed(self,smiles):
        if self.smiles is not None or smiles is None: 
            smiles =  self.smiles
            curated_smiles = self.curated_smiles
        else:
            #smiles = smiles
            curated_smiles = f'curated_{smiles}'
        return curated_smiles, smiles
        """ This function checks if a smile is passed """
    
    def remove_invalid_smiles(self,df:pd.DataFrame,smiles=None):
        _, smiles = self.is_smiles_passed(smiles)
        #df = pd.read_csv(smiles)
        removed=[]
        for i in df.index:
            try:
                smiles = df[smiles][i]
                m = Chem.MolFromSmiles(smiles)
            except:
                removed.append(i)
                #df.drop(i, inplace=True)
        df.drop(removed, inplace=True)
        #df.reset_index(drop=True, inplace=True)
        return df

    def neutralizeRadicals(self,mol):
        for a in mol.GetAtoms():
            if a.GetNumRadicalElectrons() == 1 and a.GetFormalCharge() == 1:
                a.SetNumRadicalElectrons(0)         
                a.SetFormalCharge(0)

    def table_overview(self,file_ROMol):
        if type(file_ROMol) == str:
            return file_ROMol  
        else:
            print(f"Numer of molecules: {len(file_ROMol)}")
            print(f"Numer of columns: {file_ROMol.shape[1]}")
            return file_ROMol.head(5)

    def check_extention(self,file):
        """ check the file extention and convert to ROMol if necessary """

        if file[-3:] == "sdf":       
            imported_file = PandasTools.LoadSDF(file, smilesName='SMILES', includeFingerprints=False)
            return imported_file
        
        elif file[-4:] == "xlsx":
            
            imported_file = pd.read_excel(file)
            return imported_file
            
        elif file[-3:] == "csv":
            
            imported_file = pd.read_csv(file, encoding='latin-1')   
            return imported_file
        
        else:
            return ("file extension not supported, supported extentions are: csv, xlsx and sdf")

    def rdkit_numpy_convert(self,fp):
        """Convert rdkit mol to numpy array"""
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        return np.asarray(output)

    def fp_generation(self,mols, radii, bits):

        #generates maccs fp
        maccs_fp = [MACCSkeys.GenMACCSKeys(x) for x in mols]
        maccs_fp = self.rdkit_numpy_convert(maccs_fp)

        with open('analysis/fp/maccs_fp.json', 'w', encoding='utf-8') as f:
            
            json.dump(maccs_fp, f, cls = self.NumpyArrayEncoder)

        #fcfp & ecfp fp generation
        for radius in radii:
            for bit in bits:
                ecfp_fp = [AllChem.GetMorganFingerprintAsBitVect(m, radius, bit, useFeatures=False) for m in mols]
                fcfp_fp = [AllChem.GetMorganFingerprintAsBitVect(m, radius, bit, useFeatures=True) for m in mols]

                ecfp_fp = self.rdkit_numpy_convert(ecfp_fp)
                fcfp_fp = self.rdkit_numpy_convert(fcfp_fp)

                with open('analysis/fp/ecfp_fp_{}_{}.json'.format(radius, bit), 'w', encoding='utf-8') as f:
                    json.dump(ecfp_fp, f, cls = self.NumpyArrayEncoder)
                
                with open('analysis/fp/fcfp_fp_{}_{}.json'.format(radius, bit), 'w', encoding='utf-8') as f:
                    json.dump(fcfp_fp, f, cls = self.NumpyArrayEncoder)
    
    def metal_atomic_numbers(self,at):
        """ This function checks the atomic number of an atom """
        
        n = at.GetAtomicNum()
        return (n==13) or (n>=21 and n<=31) or (n>=39 and n<=50) or (n>=57 and n<=83) or (n>=89 and n<=115)

    def is_metal(self,smile):
        """ This function checks if an atom is a metal based on its atomic number """
        mol = Chem.MolFromSmiles(smile)
        rwmol = Chem.RWMol(mol)
        rwmol.UpdatePropertyCache(strict=False)
        metal = [at.GetSymbol() for at in rwmol.GetAtoms() if self.metal_atomic_numbers(at)]
        return len(metal) == 1

    def smiles_preparator(self,smile : str or list):
        """ This function prepares smiles by removing stereochemistry """
        if type(smile) == str:
            return smile.strip("@/\\")
        if type(smile) == None:
            return "C"
        elif type(smile) == list or type(smile) == pd.core.series.Series:
            return [smile.strip("@/\\") for smile in smile]

    def remove_Salt_From_Mol(self,mol):
        """ This function removes salts, see complete list of possible salts in https://github.com/rdkit/rdkit/blob/master/Data/Salts.txt """
        salt_list = [
            None,"[Cl,Br,I]","[Li,Na,K,Ca,Mg]","[O,N]","[H]","[Ba]","[Al]","[Cu]",
            "[Cu]","[Cs]","[Zn]","[Mn]","[Sb]","[Cr]","[Ni]","Cl[Cr]Cl","[B]",
            "COS(=O)(=O)[O-]","CCN(CC)CC","NCCO","O=CO","O=S(=O)([O-])C(F)(F)F"        
            ]
        for salt in salt_list:
            remover = SaltRemover(defnData = salt)
            if salt == salt_list[0]:
                striped = remover.StripMol(mol, dontRemoveEverything = True)
            else:
                striped = remover.StripMol(striped, dontRemoveEverything = True)
        return striped
    #remove salts
    def remove_Salt_From_DF(self,df4 : pd.DataFrame, smiles=None):
        curated_smiles, smiles = self.is_smiles_passed(smiles)
        wrongSmiles = []
        new_smiles = []
        indexDropList_salts = []
        for index, smile in enumerate(df4[smiles]):
            try:
                mol = Chem.MolFromSmiles(smile)
                remov = self.remove_Salt_From_Mol(mol)
                if remov.GetNumAtoms() <= 2:
                    indexDropList_salts.append(index)
                else:
                    new_smiles.append(Chem.MolToSmiles(remov, kekuleSmiles=True))
            except:
                wrongSmiles.append(df4.iloc[[index]])
                indexDropList_salts.append(index)

        if len( indexDropList_salts ) == 0:
            df4[curated_smiles] = new_smiles 
            return df4
        else:
            #drop wrong smiles
            df4.drop(indexDropList_salts, errors = "ignore", inplace = True)
            #save removes wrong smiles
            #mask = df4.iloc[indexDropList_salts]
            #mask.to_csv("{}/error/invalid_smiles.csv".forma), sep=',', header=True, index=False)
            df4[curated_smiles] = new_smiles
            return df4
    
    def normalize_groups(self,df4 : pd.DataFrame,smiles=None):
        curated_smiles,smiles = self.is_smiles_passed(smiles)
        mols = []
        not_normalized = []
        for index,smi in enumerate(df4[smiles]):
            try:
                m = Chem.MolFromSmiles(smi,sanitize=True)
                m2 = rdMolStandardize.Normalize(m)
                smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
                mols.append(smi)
            except:
                not_normalized.append([smi,index])
        indexes = [i[1] for i in not_normalized]
        df4.drop(df4.index[indexes],inplace=True)
        df4[curated_smiles] = mols
        #normalized = pd.Series(mols)
        return df4
    #remove organometallics
    def remove_metal(self,df4 : pd.DataFrame,smiles = None):
        _,smiles = self.is_smiles_passed(smiles)
        organometals = []
        indexDropList_org = []
        
        for index, smile in enumerate(df4[smiles]):
            
            if self.is_metal(smile):
                organometals.append(df4.iloc[[index]])
                indexDropList_org.append(index)

        if len(indexDropList_org) == 0:
            return df4
        else:
            #drop organometallics
            df4 = df4.drop(df4.index[indexDropList_org])
            #save droped organometallics
            #organometal = pd.concat(organometals)
            #organmetal.to_csv("{}/error/organometallics.csv".forma), sep=',', header=True, index=False)
            return df4

    def canonical_tautomer(self,df4 : pd.DataFrame,smiles=None):
        curated_smiles,smiles = self.is_smiles_passed(smiles)
        te = rdMolStandardize.TautomerEnumerator()
        mols = []
        not_tautomerizable = []
        df5 = df4
        for index,smi in enumerate(df4[smiles]):
            try:
                m = Chem.MolFromSmiles(smi,sanitize=True)
                m2 = te.Canonicalize(m)
                smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
                mols.append(smi)
            except:
                not_tautomerizable.append([smi,index])
        indexes = [i[1] for i in not_tautomerizable]
        df4.drop(df4.index[indexes],inplace=True)
        df4[curated_smiles] = mols
        #canonical = pd.Series(mols)
        # canonical_tautomer = pd.DataFrame(mols, columns=["canonical_tautomer"])
        # df_canonical_tautomer = df.join(canonical_tautomer)
        return df4,df5

    def remove_mixture(self, df4: pd.DataFrame, smiles=None):
        """
        Splits the DataFrame into two:
        
        - df_no_mixtures (df4): Rows without mixtures (SMILES without '.')
        - df_mixtures (_): Rows with mixtures (SMILES containing '.')
        
        Parameters:
        - df4 (pd.DataFrame): Input DataFrame containing SMILES strings.
        - smiles (str, optional): Column name containing SMILES strings. If None, determined by is_smiles_passed.
        
        Returns:
        - Tuple[pd.DataFrame, Union[pd.DataFrame, str]]: (df_no_mixtures, df_mixtures or "No mixture")
        """
        
        # Determine the column names for SMILES
        curated_smiles, smiles = self.is_smiles_passed(smiles)
        
        # Initialize list to hold indices of mixtures
        indexDropList_mix = []
        
        # Make a copy of df4 for mixtures
        df5 = df4.copy()
        
        # Iterate over the SMILES column
        for index, smile in enumerate(df4[smiles]):
            try:
                if isinstance(smile, str):
                    if '.' in smile:
                        indexDropList_mix.append(index)
            except:
                pass
        
        try:
            if not indexDropList_mix:
                return df4, None, "No mixture"
            else:
                # Drop mixtures from df4 to create df_no_mixtures
                df_no_mixtures = df4.drop(indexDropList_mix).copy()
                
                # Select mixtures into df_mixtures from df5
                df_mixtures = df5.iloc[indexDropList_mix].copy()
                
                return df_no_mixtures, df_mixtures, "No mixture"
        except:
            return df4, None, "No mixture"
        
    def neutralize(self,df4 : pd.DataFrame, smiles=None,):
        curated_smiles, smiles = self.is_smiles_passed(smiles)
        df5 = df4.copy()
        mols_noradical = []
        standAlone_salts = []
        indexDropList_salts = []
        for index, smile in enumerate(df4[smiles]):
            try:
                m = Chem.MolFromSmiles(smile, sanitize=True)
                m = rd_inchi.MolToInchi(m)
                m = Chem.MolFromInchi(m)
                self.neutralizeRadicals(m)
                Chem.SanitizeMol(m)
                mols_noradical.append(Chem.MolToSmiles(m, kekuleSmiles=True))
            except:
                indexDropList_salts.append(index)
                standAlone_salts.append(df4.iloc[[index]])
        if len(standAlone_salts) == 0:
            return df4,df5
        else:
            df4 = df4.drop(df4.index[indexDropList_salts])
            #salts = pd.Series(standAlone_salts)
            #salts.to_csv("{}/error/salts.csv".forma), sep=',', header = True, index = False)
            df4[curated_smiles] = mols_noradical
            return df4,df5 #, salts

    def standardise(self, df4: pd.DataFrame, smiles=None):
        curated_smiles, smiles = self.is_smiles_passed(smiles)
        
        # Create a copy of the DataFrame to preserve original data
        df5 = df4.copy()
        
        # Define a function to standardize SMILES
        def standardize_smiles(smile):
            try:
                standardized_smile = rdMolStandardize.StandardizeSmiles(smile)
                return standardized_smile
            except:
                return None
        
        # Apply the standardization function to each SMILES string
        df4[curated_smiles] = df4[smiles].apply(standardize_smiles)
        
        return df4, df5
        
    def std_routine(self, df4 : pd.DataFrame, smiles=None):
        curated_smiles, smiles = self.is_smiles_passed(smiles)
        if smiles.find("curated_") == -1:
            curated_smiles = smiles
        
        df4 = self.remove_invalid_smiles(df4, smiles)
        df4 = self.remove_Salt_From_DF(df4, smiles)
        df4 = self.remove_metal(df4, curated_smiles)
        df4,_ = self.remove_mixture(df4, curated_smiles)
        df4 = self.normalize_groups(df4,curated_smiles)
        df4,_ = self.neutralize(df4, curated_smiles)
        df4,_ = self.canonical_tautomer(df4, curated_smiles)
        df4,_ = self.standardise(df4, curated_smiles)
        
        #remove salts second time
        #why cant we just use the function?
        
      
        #remove radicals and standalone salts
        mols_noradical = []
        standAlone_salts = []
        indexDropList_salts = []
        for index, smile in enumerate(df4[curated_smiles]):
            try:
                m = Chem.MolFromSmiles(smile, False)
                m = rd_inchi.MolToInchi(m)
                m = Chem.MolFromInchi(m)
                self.neutralizeRadicals(m)
                Chem.SanitizeMol(m)
                mols_noradical.append(Chem.MolToSmiles(m, False))
            except:
                indexDropList_salts.append(index)
                standAlone_salts.append(df4.iloc[[index]])
        if len(standAlone_salts) == 0:
            pass
        else:
            df4 = df4.drop(df4.index[indexDropList_salts])
            salts = pd.concat(standAlone_salts)
            #salts.to_csv("{}/error/salts.csv".forma), sep=',', header = True, index = False)
        df4[curated_smiles] = mols_noradical

        #remove salts second time
        df4 = self.remove_Salt_From_DF(df4, curated_smiles)
        df4 = self.remove_metal(df4, curated_smiles)
        df4 , _ = self.remove_mixture(df4, curated_smiles)
        df4 , _ = self.neutralize(df4, curated_smiles)
        #final std
        rdMol = [Chem.MolFromSmiles(smile, sanitize = True) for smile in df4[curated_smiles]]

        molBlock = [Chem.MolToMolBlock(mol) for mol in rdMol]

        stdMolBlock = [standardizer.standardize_molblock(mol_block) for mol_block in molBlock]

        molFromMolBlock = [Chem.MolFromMolBlock(std_molblock) for std_molblock in stdMolBlock]

        mol2smiles = [Chem.MolToSmiles(m) for m in molFromMolBlock]
        

        #remove unwanted columns
        # dropList = ['SMILES', 'SMILES_no_stereo', 'SMILES_no_salts', 'Stand_smiles', 'SMILES_salts_removed_1', curated_smiles, 'SMILES_salts_removed_2']
        # df4 = df4.drop(columns = dropList)
        df4['ROMol'] = molFromMolBlock
        df4[curated_smiles] = mol2smiles
        return df4
