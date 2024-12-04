#if "updated_df" not in st.session_state:
    #         st.session_state.updated_df = df
        
    #         st.header('**Original input data**')
    #         AgGrid(df)
   
    #     st.sidebar.header("Please delete undesired columns")
        
    #     with st.sidebar.form("my_form"):
    #         index = df.columns.tolist().index(
    #             st.session_state["updated_df"].columns.tolist()[0]
    #         )
    #         st.selectbox(
    #             "Select column to delete", options=df.columns, index=index, key="delete_col"
    #         )
    #         delete = st.form_submit_button(label="Delete")
    #     if delete:
    #         persist_dataframe("updated_df","delete_col")
    # else:
    #     st.info('Awaiting for CSV file to be uploaded.')
# with st.sidebar.header('2. Upload your CSV data (calculated descriptors)'):
#         uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#     st.sidebar.markdown("""
#     [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/descriptor_morgan_r2_2048bits_for_modeling.csv)
#     """)

#     # Read Uploaded file and convert to pandas
#     if uploaded_file is not None:
#         # Read CSV data
#         df = pd.read_csv(uploaded_file, sep=',')

#         st.header('**Molecular descriptors input data**')

#         AgGrid(df)

#     else:
#         st.info('Awaiting for CSV file to be uploaded.')

#     st.sidebar.write('---')
# with st.sidebar.header('2. Upload your CSV data (calculated descriptors)'):
#         uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#     st.sidebar.markdown("""
#     [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/descriptor_morgan_r2_2048bits_for_modeling.csv)
#     """)

#     # Read Uploaded file and convert to pandas
#     if uploaded_file is not None:
#         # Read CSV data
#         df = pd.read_csv(uploaded_file, sep=',')

#         st.header('**Molecular descriptors input data**')

#         AgGrid(df)

#     else:
#         st.info('Awaiting for CSV file to be uploaded.')

#     st.sidebar.write('---')
  # n_estimator_container=st.sidebar.container()
            # min_parameter_n_estimators = n_estimator_container.number_input('Minimal value of estimators (n_estimators)', 50,key="min_parameter_n_estimators", max_value=None, step=1)
            # if "max_parameter_n_estimators" in s_state:
            #     if s_state["max_parameter_n_estimators"] <= min_parameter_n_estimators:
            #         s_state["max_parameter_n_estimators"] = min_parameter_n_estimators+1
            #     else:
            #         pass
            #     max_parameter_n_estimators = n_estimator_container.number_input('Maximum value of estimators (n_estimators)', s_state["max_parameter_n_estimators"], key="max_parameter_n_estimators", max_value=None, step=1)
            # else:
            #     max_parameter_n_estimators = n_estimator_container.number_input('Maximum value of estimators (n_estimators)', min_parameter_n_estimators+1, key="max_parameter_n_estimators", max_value=None, step=1)
            #     n_estimator_container.write("First value (minimum) must be smaller than second(maximum) value")
#min_parameter_max_depth = st.sidebar.number_input('Minimum Max depth (max_depth)', 1, 200,key="min_parameter_max_depth")
            # #ensure that the max depth is not smaller than the min depth
            # if "max_parameter_max_depth" in s_state:
            #     if s_state["max_parameter_max_depth"] <= min_parameter_max_depth:
            #         s_state["max_parameter_max_depth"] = min_parameter_max_depth+1
            #     else:
            #         pass
            #     max_parameter_max_depth = st.sidebar.number_input('Maximum Max depth (max_depth)', s_state["max_parameter_max_depth"], 200,key="max_parameter_max_depth")
            # else:
            #     max_parameter_max_depth = st.sidebar.number_input('Maximum Max depth (max_depth)', min_parameter_max_depth+1, 200,key="max_parameter_max_depth")
            #     s_state["max_parameter_max_depth"] = max_parameter_max_depth

            # max_depth = {"max_depth": [min_parameter_max_depth, max_parameter_max_depth]}
            # selected_hyperparameters.update(max_depth)
            # #st.write(selected_hyperparameters)
            #min_parameter_min_samples_split = st.sidebar.number_input('Minimum number of samples required to split an internal node (min_samples_split)', 2, 50)
            # max_parameter_min_samples_split = st.sidebar.number_input('Maximum number of samples required to split an internal node (min_samples_split)', min_parameter_min_samples_split+1, 50)
            # min_samples_split = {'min_samples_split': [min_parameter_min_samples_split, max_parameter_min_samples_split]}
            # selected_hyperparameters.update(min_samples_split)
            #min_parameter_min_samples_leaf = st.sidebar.number_input('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 50)
            # max_parameter_min_samples_leaf = st.sidebar.number_input('Maximum number of samples required to be at a leaf node (min_samples_leaf)', min_parameter_min_samples_leaf, 50)
            # min_samples_leaf = {'min_samples_leaf': [min_parameter_min_samples_leaf,max_parameter_min_samples_leaf]}
            # selected_hyperparameters.update(min_samples_leaf)
            # #Generate Image from Neutralized SMILES
                # PandasTools.AddMoleculeColumnToFrame(not_canon, smilesCol = curate.curated_smiles,
                # molCol = "No_mixture", includeFingerprints = False)
                # #Generate Image from No_mixture SMILES
                # PandasTools.AddMoleculeColumnToFrame(canonical_tautomer, smilesCol=curate.curated_smiles,
                # molCol = "Canonical_tautomer", includeFingerprints = False)
                # # Filter only columns containing images
                # canonical_tautomer_fig = canonical_tautomer.filter(items = ["No_mixture", "Canonical_tautomer"])
                # # Show table for comparing
                # st.write(canonical_tautomer_fig.to_html(escape = False), unsafe_allow_html = True)