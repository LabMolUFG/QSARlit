
########################################################################################################################################
# Credits
########################################################################################################################################

# Developed by José Teófilo Moreira Filho, Ph.D.
# teofarma1@gmail.com
# http://lattes.cnpq.br/3464351249761623
# https://www.researchgate.net/profile/Jose-Teofilo-Filho
# https://scholar.google.com/citations?user=0I1GiOsAAAAJ&hl=pt-BR
# https://orcid.org/0000-0002-0777-280X

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
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from st_aggrid import AgGrid
import utils
def app(df,s_state):

    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
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
    
    def remove_invalid(df):
        for i in df.index:
            try:
                smiles = df[name_smiles][i]
                m = Chem.MolFromSmiles(smiles)
            except:
                df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    ##################################################################
    # def remove_metals(df):
    #     badAtoms = Chem.MolFromSmarts('[!$([#1,#3,#11,#19,#4,#12,#20,#5,#6,#14,#7,#15,#8,#16,#9,#17,#35,#53])]')
    #     mols = []
    #     for i in df.index:
    #         smiles = df[name_smiles][i]
    #         m = Chem.MolFromSmiles(smiles,)
    #         if m.HasSubstructMatch(badAtoms):
    #             df.drop(i, inplace=True)
    #     df.reset_index(drop=True, inplace=True)
    #     return df
    # ##################################################################
    # def normalize_groups(df):
    #     mols = []
    #     for smi in df[name_smiles]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = rdMolStandardize.Normalize(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     norm = pd.DataFrame(mols, columns=["normalized_smiles"])
    #     df_normalized = df.join(norm)
    #     return df_normalized
    # ##################################################################
    # def neutralize(df):
    #     uncharger = rdMolStandardize.Uncharger()
    #     mols = []
    #     for smi in df['normalized_smiles']:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = uncharger.uncharge(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     neutral = pd.DataFrame(mols, columns=["neutralized_smiles"])
    #     df_neutral = df.join(neutral)
    #     return df_neutral
    # ##################################################################
    # def no_mixture(df):
    #     mols = []
    #     for smi in df["neutralized_smiles"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = rdMolStandardize.FragmentParent(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     no_mixture = pd.DataFrame(mols, columns=["no_mixture_smiles"])
    #     df_no_mixture = df.join(no_mixture)
    #     return df_no_mixture
    # ##################################################################
    # def canonical_tautomer(df):
    #     te = rdMolStandardize.TautomerEnumerator()
    #     mols = []
    #     for smi in df["no_mixture_smiles"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = te.Canonicalize(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     canonical_tautomer = pd.DataFrame(mols, columns=["canonical_tautomer"])
    #     df_canonical_tautomer = df.join(canonical_tautomer)
    #     return df_canonical_tautomer
    # ##################################################################
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

    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File
    
    #st.header('**Original input data**')

    # Read Uploaded file and convert to pandas
    if df is not None:
        # Read CSV data
        #df = pd.read_csv(uploaded_file, sep=',')

        st.sidebar.write('---')

        # Select columns
        with st.sidebar.header('1. Enter column names'):
            name_smiles = st.sidebar.selectbox('Select column containing SMILES', options=df.columns, key="smiles_column")
            # name_activity = st.sidebar.selectbox(
            #     'Select column containing Activity (Active and Inactive should be 1 and 0, respectively or numerical values)', 
            #     options=df.columns, key="outcome_column"
            #     )
            curate = utils.Curation(name_smiles)
        ########################################################################################################################################
        # Sidebar - Select visual inspection
        ########################################################################################################################################

        st.sidebar.header('2. Visual inspection')

        st.sidebar.subheader('Select step for visual inspection')
                
        container = st.sidebar.container()
        _all = st.sidebar.checkbox("Select all")
        
        options=['Normalization',
                'Neutralization',
                'Mixture_removal',
                'Canonical_tautomers']
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
            remove_invalid(df)

            #---------------------------------------------------------------------------------#
            # Remove compounds with metals
            curate.remove_metals(df)

            #---------------------------------------------------------------------------------#
            # Normalize groups

            if options[0] in selected_options:

                st.header('**Normalized Groups**')

                normalized = curate.normalize_groups(df)

                #Generate Image from original SMILES
                PandasTools.AddMoleculeColumnToFrame(normalized, smilesCol=name_smiles,
                molCol='Original', includeFingerprints=False)
                #Generate Image from normalized SMILES
                PandasTools.AddMoleculeColumnToFrame(normalized, smilesCol="normalized_smiles",
                molCol='Normalized', includeFingerprints=False)
                # Filter only columns containing images
                normalized_fig = normalized.filter(items=['Original', "Normalized"])
                    # Show table for comparing
                st.write(normalized_fig.to_html(escape=False), unsafe_allow_html=True)

            else:
                normalized = curate.normalize_groups(df)
                #redundante?
            #----------------------------------------------------------------------------------#
            # Neutralize when possible
            if options[1] in selected_options:

                st.header('**Neutralized Groups**')
                #if options[0] in selected_options:
                neutralized = curate.neutralize(normalized)
                # else:
                #     neutralized=neutralize(df)
                #Generate Image from normalized SMILES
                PandasTools.AddMoleculeColumnToFrame(neutralized, smilesCol="normalized_smiles",
                molCol="Normalized", includeFingerprints=False)
                #Generate Image from Neutralized SMILES
                PandasTools.AddMoleculeColumnToFrame(neutralized, smilesCol="neutralized_smiles",
                molCol="Neutralized", includeFingerprints=False)
                # Filter only columns containing images
                neutralized_fig = neutralized.filter(items=["Normalized", "Neutralized"])
                # Show table for comparing
                st.write(neutralized_fig.to_html(escape=False), unsafe_allow_html=True)

            else:
                neutralized = curate.neutralize(normalized)

            #---------------------------------------------------------------------------------#
            # Remove mixtures and salts
            if options[2] in selected_options:

                st.header('**Remove mixtures**')
                # if options[1] in selected_options:
                no_mixture = curate.no_mixture(neutralized)
            
                #Generate Image from Neutralized SMILES
                PandasTools.AddMoleculeColumnToFrame(no_mixture, smilesCol="neutralized_smiles",
                molCol="Neutralized", includeFingerprints=False)
                #Generate Image from No_mixture SMILES
                PandasTools.AddMoleculeColumnToFrame(no_mixture, smilesCol="no_mixture_smiles",
                molCol="No_mixture", includeFingerprints=False)
                # Filter only columns containing images
                no_mixture_fig = no_mixture.filter(items=["Neutralized", "No_mixture"])
                # Show table for comparing
                st.write(no_mixture_fig.to_html(escape=False), unsafe_allow_html=True)
            else:
                no_mixture = curate.no_mixture(neutralized)

            #---------------------------------------------------------------------------------#

            #Generate canonical tautomers
            if options[3] in selected_options:

                st.header('**Generate canonical tautomers**')
                # if options[2] in selected_options:
                canonical_tautomer = curate.canonical_tautomer(no_mixture)
                #Generate Image from Neutralized SMILES
                PandasTools.AddMoleculeColumnToFrame(canonical_tautomer, smilesCol="no_mixture_smiles",
                molCol="No_mixture", includeFingerprints=False)
                #Generate Image from No_mixture SMILES
                PandasTools.AddMoleculeColumnToFrame(canonical_tautomer, smilesCol="canonical_tautomer",
                molCol="Canonical_tautomer", includeFingerprints=False)
                # Filter only columns containing images
                canonical_tautomer_fig = canonical_tautomer.filter(items=["No_mixture", "Canonical_tautomer"])
                # Show table for comparing
                st.write(canonical_tautomer_fig.to_html(escape=False), unsafe_allow_html=True)

            else:
                canonical_tautomer = curate.canonical_tautomer(no_mixture)

        ########################################################################################################################################
        # Analysis of duplicates
        ########################################################################################################################################
                    
            
            filedownload(canonical_tautomer,"Standardized with Duplicates")
        
        #--------------------------- Removal of duplicates------------------------------#
            # Generate InchiKey
            inchikey = curate.smi_to_inchikey(canonical_tautomer)

            no_dup = inchikey.drop_duplicates(subset='inchikey', keep="first")


        #--------------------------- Print dataframe without duplicates------------------------------#

            st.header('**Duplicates removed**')
            #Keep only curated smiles and outcome
            no_dup = no_dup.filter(items=["canonical_tautomer",])
            no_dup.rename(columns={"canonical_tautomer": "SMILES",},inplace=True)
            no_dup = no_dup.join(st.session_state.updated_df.drop(name_smiles, 1))
            # Display curated dataset
            AgGrid(no_dup)

        ########################################################################################################################################
        # Data download
        ########################################################################################################################################

            # File download
            filedownload(no_dup,"Curated")
