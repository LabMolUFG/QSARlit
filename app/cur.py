########################################################################################################################################
# Importing packages
########################################################################################################################################

import streamlit as st

import base64
import warnings
warnings.filterwarnings(action='ignore')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import pandas as pd

from rdkit.Chem import PandasTools
#from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
import utils
from rdkit.Chem.MolStandardize import rdMolStandardize

from st_aggrid import AgGrid
def app(df,s_state):
    curated_key = utils.Commons().CURATED_DF_KEY
    #custom = cur.Custom_Components()
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    cc = utils.Custom_Components()
    def persist_dataframe(updated_df,col_to_delete):
            # drop column from dataframe
            delete_col = st.session_state[col_to_delete]

            if delete_col in st.session_state[updated_df]:
                st.session_state[updated_df] = st.session_state[updated_df].drop(columns=[delete_col])
            else:
                st.sidebar.warning("Column previously deleted. Select another column.")
            with st.container():
                st.header("**Updated input data**") 
                AgGrid(st.session_state[updated_df])
                st.header('**Original input data**')
                AgGrid(df)

    def filedownload(df,data):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            st.header(f"**Download {data} data**")
            href = f'<a href="data:file/csv;base64,{b64}" download="{data}_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def remove_invalid(df,smiles_col):
        for i in df.index:
            try:
                smiles = df[smiles_col][i]
                m = Chem.MolFromSmiles(smiles)
            except:
                df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    # def smi_to_inchikey(df):
    #     inchi = []
    #     for smi in df["canonical_tautomer"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = Chem.inchi.MolToInchiKey(m)
    #         inchi.append(m2)
    #     inchikey = pd.DataFrame(inchi, columns=["inchikey"])
    #     df_inchikey = df.join(inchikey)
    #     return df_inchikey
    # ##################################################################
    def file_dependent_code(df):

        # Select columns
        with st.sidebar.header('1. Select column names'):
            name_smiles = st.sidebar.selectbox('Select column containing SMILES', options=df.columns, key="smiles_column")
            name_activity = st.sidebar.selectbox(
                'Select column containing Activity (Active and Inactive should be 1 and 0, respectively or numerical values)', 
                options=df.columns, key="outcome_column"
                )
        curate = utils.Curation(name_smiles)
        st.sidebar.write('---')

        ########################################################################################################################################
        # Sidebar - Select visual inspection
        ########################################################################################################################################

        st.sidebar.header('2. Curation steps')

        st.sidebar.subheader('Select step for visual inspection')
                
        container = st.sidebar.container()
        _all = st.sidebar.checkbox("Select all")
        data_type = ["Continuous", "Categorical"]
        radio = st.sidebar.radio(
        "Continuous or categorical activity?",
        data_type, key="activity_type",horizontal=True 
        )
        
        options=['Normalization',
                'Neutralization',
                'Mixture_removal',
                'Canonical_tautomers',
                'Chembl_Standardization',
                ]
        if _all:
            selected_options = container.multiselect("Select one or more options:", options, options)
        else:
            selected_options =  container.multiselect("Select one or more options:", options)


        ########################################################################################################################################
        # Apply standardization
        ########################################################################################################################################

        if st.sidebar.button('Standardize'):

            #---------------------------------------------------------------------------------#
            # Remove invalid smiles
            remove_invalid(df,name_smiles)
            df[name_smiles] = curate.smiles_preparator(df[name_smiles])
            st.header("1. Invalid SMILES removed")
            cc.AgGrid(df,key = "invalid_smiles_removed")
            #---------------------------------------------------------------------------------#
            # Remove compounds with metals
            df = curate.remove_Salt_From_DF(df, name_smiles)
            df = curate.remove_metal(df, name_smiles)
            normalized = curate.normalize_groups(df)
            neutralized,_ = curate.neutralize(normalized,curate.curated_smiles)
            no_mixture, mixtures, only_mixture = curate.remove_mixture(neutralized,curate.curated_smiles)
            canonical_tautomer,_ = curate.canonical_tautomer(no_mixture,curate.curated_smiles)
            standardized,_= curate.standardise(canonical_tautomer,curate.curated_smiles)
            #---------------------------------------------------------------------------------#
            # Remove mixtures and salts
            if options[2] in selected_options:

                st.header('**Remove mixtures**')
                # if options[1] in selected_options:
                with st.container():
                    #st.header("Mixture")
                    if only_mixture=="No mixture":
                        st.write("**No mixture found**")
                    else:
                        cc.img_AgGrid(only_mixture,title = "Mixture",mol_col=curate.curated_smiles,key="mixture")
        #---------------------------------------------------------------------------------#
            
        ########################################################################################################################################
        # Download Standardized with Duplicates
        ########################################################################################################################################
                
            # std_with_dup = canonical_tautomer.filter(items=["canonical_tautomer",])
            # std_with_dup.rename(columns={"canonical_tautomer": "SMILES",},inplace=True)
            # std_with_dup = std_with_dup.join(st.session_state.updated_df.drop(name_smiles, 1))

            filedownload(standardized,"Standardized with Duplicates")
            def duplicate_analysis(df,dups):
                try:
                    st.header('**Duplicate Analysis**')
                    st.write("Number of duplicates removed: ",dups)
                    st.write("Number of compounds remaining: ",len(df))
                    st.write("Percentage of compounds removed: ",round(dups/len(df)*100,2),"%")
                    st.write("Percentage of compounds remaining: ",round(100-dups/len(df)*100,2),"%")
                except:
                    st.write("No duplicates found!")
                #st.header("**Final Dataset**")
                df.drop_duplicates(subset=curate.curated_smiles,keep="first",inplace=True)
                filedownload(df,"Final Dataset")
            if radio == data_type[0]:
                continuous = utils.Continuous_Duplicate_Remover(standardized,curate.curated_smiles,name_activity,False,False)
                continuous,dups = continuous.remove_duplicates()
                s_state[curated_key] = continuous
                duplicate_analysis(continuous,dups)

            elif radio == data_type[1]:
                categorical = utils.Classification_Duplicate_Remover(standardized,curate.curated_smiles,name_activity)
                categorical,dups = categorical.remove_duplicates()
                s_state[curated_key] = categorical
                duplicate_analysis(categorical,dups)

    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################
    # Upload File
    # df = custom.upload_file()
    st.write('---')

    #st.header('**Original input data**')

    # Read Uploaded file and convert to pandas
    if df is None:
        st.warning("No dataset loaded. Please upload a dataset.")
        uploaded_file = st.file_uploader("Upload your CSV data", type=['csv'], key='data_upload')
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                s_state.df = df  # Save df in session state
                st.success('Data uploaded successfully')
                st.header("**Data Preview**")
                st.write(df.head())
                file_dependent_code(df)
            except Exception as e:
                st.error(f"An error occurred while loading the data: {e}")
    else:
        st.header("**Data Preview**")
        st.write(df.head())
        file_dependent_code(df)

