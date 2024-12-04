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

from st_aggrid import AgGrid
import streamlit as st

import base64
import warnings
warnings.filterwarnings(action='ignore')

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

import numpy as np

import pandas as pd
import utils

def app(df, s_state):
    cc = utils.Custom_Components()
    if utils.Commons().CURATED_DF_KEY in s_state:
        df = s_state[utils.Commons().CURATED_DF_KEY]
    else:
        df = None  # Ensure df is defined

    st.title("Calculate Descriptors")

    ########################################################################################################################################
    # Functions
    ########################################################################################################################################

    def smiles_to_fp(mols, radius, nbits, method):
        fps = []
        for mol in mols:
            mol_obj = Chem.MolFromSmiles(mol)
            if mol_obj is not None:
                if method == 'Morgan':
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol_obj, radius, nBits=nbits)
                elif method == 'MACCS':
                    fp = MACCSkeys.GenMACCSKeys(mol_obj)
                fps.append(fp)
            else:
                fps.append(None)
        return fps

    def fingerprints_to_dataframe(fps, nbits, method):
        fps_array = []
        for fp in fps:
            if fp is not None:
                if method == 'MACCS':
                    arr = np.zeros((166,), dtype=int)
                else:
                    arr = np.zeros((nbits,), dtype=int)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps_array.append(arr)
            else:
                # Handle invalid molecules
                if method == 'MACCS':
                    fps_array.append([np.nan] * 166)
                else:
                    fps_array.append([np.nan] * nbits)
        # Create DataFrame from fingerprints
        if method == 'MACCS':
            fps_df = pd.DataFrame(fps_array)
            fps_df.columns = [f'Bit_{i}' for i in range(1, 167)]
        else:
            fps_df = pd.DataFrame(fps_array)
            fps_df.columns = [f'Bit_{i}' for i in range(1, nbits + 1)]
        return fps_df

    def filedownload(dataframe, filename):
        csv = dataframe.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

    ########################################################################################################################################
    # Main Logic
    ########################################################################################################################################

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
            except Exception as e:
                st.error(f"An error occurred while loading the data: {e}")
                return
    else:
        st.header("**Data Preview**")
        st.write(df.head())

    if df is not None:
        # Select columns
        with st.sidebar.header('1. Select Column Names'):
            name_smiles = st.sidebar.selectbox('Select column containing SMILES', df.columns)

        # Select fingerprint method and parameters
        method = 'Morgan'
        if method == 'Morgan':
            st.sidebar.subheader('Morgan Parameters')
            radius = st.sidebar.number_input('Enter the radius', min_value=1, max_value=10, value=2)
            nbits = st.sidebar.number_input('Enter the number of bits', min_value=64, max_value=8192, value=2048)
        elif method == 'MACCS':
            st.sidebar.write('MACCS keys have a fixed size of 166 bits.')
            nbits = 166  # MACCS keys have 166 bits

        ########################################################################################################################################
        # Apply descriptor calculation
        ########################################################################################################################################

        if st.sidebar.button('Calculate Descriptors'):
            try:
                df_smiles = df.copy()
                # Calculate descriptors
                descriptors = smiles_to_fp(
                    mols=df_smiles[name_smiles],
                    radius=radius if method == 'Morgan' else None,
                    nbits=nbits if method == 'Morgan' else None,
                    method=method
                )

                # Convert descriptors to DataFrame
                fps_df = fingerprints_to_dataframe(descriptors, nbits, method)

                # Combine with original DataFrame
                df_combined = pd.concat([df_smiles.reset_index(drop=True), fps_df], axis=1)

                # Display calculated descriptors
                st.header("**Calculated Descriptors**")
                cc.AgGrid(df_combined, "Calculated descriptors")

                ########################################################################################################################################
                # Descriptor download
                ########################################################################################################################################

                st.header("**Download Descriptors**")
                filename = f'descriptor_{method}_{nbits}bits.csv'
                filedownload(df_combined, filename)

            except Exception as e:
                st.error(f"An error occurred during descriptor calculation: {e}")
    else:
        st.info("Please upload a dataset to proceed.")
