import streamlit as st
from st_aggrid import AgGrid

import base64
from io import BytesIO
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import io
from numpy import sqrt
from numpy import argmax

import pandas as pd

import matplotlib.pyplot as plt

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, roc_curve, roc_auc_score, make_scorer
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
import pickle
from sklearn.calibration import calibration_curve

from imblearn.metrics import geometric_mean_score

from skopt import BayesSearchCV

import plotly.graph_objects as go

def app(df, s_state):
    ########################################################################################################################################
    # Initialize session state variables if they don't exist
    ########################################################################################################################################
    if 'modeling_done' not in s_state:
        s_state['modeling_done'] = False

    if 'columns_to_delete' not in s_state:
        s_state['columns_to_delete'] = []

    if 'df_original' not in s_state:
        if df is not None:
            s_state['df_original'] = df.copy()
            s_state['df'] = df.copy()
        else:
            s_state['df_original'] = None
            s_state['df'] = None
    else:
        df = s_state['df']

    if 'parameter_random_state' not in s_state:
        s_state['parameter_random_state'] = 42

    if 'parameter_n_iter' not in s_state:
        s_state['parameter_n_iter'] = 10  # default value

    if 'parameter_n_jobs' not in s_state:
        s_state['parameter_n_jobs'] = -1  # default value

    if 'selected_options' not in s_state:
        s_state['selected_options'] = []

    if 'selected_hyperparameters' not in s_state:
        s_state['selected_hyperparameters'] = {}

    ########################################################################################################################################
    # Sidebar
    ########################################################################################################################################
    with st.sidebar.header('1. Set seed for reproducibility'):
        s_state['parameter_random_state'] = st.sidebar.number_input('Seed number (random_state)', value=s_state['parameter_random_state'], step=1, key='random_state')

    ########################################################################################################################################
    # File Upload and Dataset Handling
    ########################################################################################################################################

    if df is None:
        st.write('No dataset loaded. Please upload a dataset.')
        # Provide file uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            s_state['df'] = df.copy()
            s_state['df_original'] = df.copy()
        else:
            st.stop()  # Stop the script here until a file is uploaded
    else:
        st.write('Below is the loaded dataset:')

    ########################################################################################################################################
    # Select columns to delete
    ########################################################################################################################################
    st.subheader('Select columns to delete from the dataset')

    with st.form('delete_columns_form'):
        # Get list of columns
        columns = df.columns.tolist()

        # Ensure default values are valid
        default_columns_to_delete = [col for col in s_state.get('columns_to_delete', []) if col in columns]

        # Use st.multiselect to select columns to delete
        columns_to_delete = st.multiselect('Select columns to delete', columns, default=default_columns_to_delete, key='columns_to_delete_widget')

        # Delete columns button
        delete_columns_button = st.form_submit_button('Delete Selected Columns')

    if delete_columns_button:
        # Update s_state['columns_to_delete']
        s_state['columns_to_delete'] = columns_to_delete

        # Delete selected columns from df
        if columns_to_delete:
            df = df.drop(columns=columns_to_delete)
            s_state['df'] = df
            st.write('Updated dataset after deleting columns:')
            st.write(df)
        else:
            st.write('No columns selected for deletion.')
    else:
        df = s_state['df']

    ########################################################################################################################################
    # Select activity column
    ########################################################################################################################################
    with st.sidebar.header('2. Select column with activity'):
        # Ensure default value is valid
        if 'name_activity' in s_state and s_state['name_activity'] in df.columns:
            index = df.columns.tolist().index(s_state['name_activity'])
        else:
            index = 0  # or any other default index
            s_state['name_activity'] = df.columns[index]

        name_activity = st.sidebar.selectbox(
            'Select the column with activity (e.g., Active and Inactive that should be 1 and 0, respectively)',
            df.columns, index=index, key='name_activity')

        if len(name_activity) > 0:
            if name_activity not in df.columns:
                st.error(f"The column '{name_activity}' is not in the dataframe.")
    st.sidebar.write('---')

    ########################################################################################################################################
    # Data splitting
    ########################################################################################################################################

    # Selecting x and y from input file
    selected_splitting = True

    if selected_splitting == True:
        x = df.drop(columns=[name_activity]).values  # All columns except the activity column
        y = df[name_activity].values  # The activity column


    ########################################################################################################################################
    # Sidebar - Specify parameter settings
    ########################################################################################################################################

    st.sidebar.header('5. Set Parameters - Bayesian hyperparameter search')

    # Choose the general hyperparameters
    st.sidebar.subheader('General Parameters')

    s_state['parameter_n_iter'] = st.sidebar.slider('Number of iterations (n_iter)', 1, 1000, s_state['parameter_n_iter'], 1, key='n_iter')

    st.sidebar.write('---')
    s_state['parameter_n_jobs'] = st.sidebar.selectbox('Number of jobs to run in parallel (n_jobs)', options=[-1, 1], index=0, key='n_jobs')

    # Choose the hyperparameters intervals to be tested
    st.sidebar.subheader('Learning Hyperparameters')

    hyperparameters = {
        'max_features': ['sqrt'],
        'n_estimators': [100, 1000],
        'max_depth': [2, 100],
        'min_samples_leaf': [1,20], 
        'min_samples_split': [2, 20]
    }


    ########################################################################################################################################
    # Modeling
    ########################################################################################################################################

    run_modeling = st.sidebar.button('Run Modeling', key='run_modeling')

    if run_modeling or s_state['modeling_done']:
        if not s_state['modeling_done']:
            try:
                s_state['modeling_done'] = True

                ########################################################################################################################################
                # Define functions used in modeling
                ########################################################################################################################################

                def getNeighborsDitance(trainingSet, testInstance, k):
                    neighbors_k = metrics.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
                    neighbors_k.sort(0)
                    similarity = 1 - neighbors_k
                    return similarity[k - 1, :]

                # Cross-validation function
                def cros_val(x, y, classifier, cv):
                    probs_classes = []
                    y_test_all = []
                    AD_fold = []
                    distance_train_set = []
                    distance_test_set = []
                    y_pred_ad = []
                    y_exp_ad = []
                    for train_index, test_index in cv.split(x, y):
                        clf = classifier  # model with best parameters
                        X_train_folds = x[train_index]  # descriptors train split
                        y_train_folds = np.array(y)[train_index.astype(int)]  # label train split
                        X_test_fold = x[test_index]  # descriptors test split
                        y_test_fold = np.array(y)[test_index.astype(int)]  # label test split
                        clf.fit(X_train_folds, y_train_folds)  # train fold
                        y_pred = clf.predict_proba(X_test_fold)  # test fold
                        probs_classes.append(y_pred)  # all predictions for test folds
                        y_test_all.append(y_test_fold)  # all folds' labels
                        k = int(round(pow((len(y)), 1.0 / 3), 0))
                        distance_train = getNeighborsDitance(X_train_folds, X_train_folds, k)
                        distance_train_set.append(distance_train)
                        distance_test = getNeighborsDitance(X_train_folds, X_test_fold, k)
                        distance_test_set.append(distance_test)
                        Dc = np.average(distance_train) - (0.5 * np.std(distance_train))
                        for i in range(len(X_test_fold)):
                            ad = 0
                            if distance_test_set[0][i] >= Dc:
                                ad = 1
                            AD_fold.append(ad)
                    probs_classes = np.concatenate(probs_classes)
                    y_experimental = np.concatenate(y_test_all)
                    # Uncalibrated model predictions
                    pred = (probs_classes[:, 1] > 0.5).astype(int)
                    for i in range(len(AD_fold)):
                        if AD_fold[i] == 1:
                            y_pred_ad.append(pred[i])
                            y_exp_ad.append(y_experimental[i])

                    return pred, y_experimental, probs_classes, AD_fold, y_pred_ad, y_exp_ad

                # STATISTICS
                def calc_statistics(y, pred):
                    # save confusion matrix and slice into four pieces
                    confusion = confusion_matrix(y, pred)
                    # [row, column]
                    TP = confusion[1, 1]
                    TN = confusion[0, 0]
                    FP = confusion[0, 1]
                    FN = confusion[1, 0]

                    # calc statistics
                    accuracy = round(accuracy_score(y, pred), 2)  # accuracy
                    mcc = round(matthews_corrcoef(y, pred), 2)  # mcc
                    kappa = round(cohen_kappa_score(y, pred), 2)  # kappa
                    sensitivity = round(recall_score(y, pred), 2)  # Sensitivity
                    specificity = round(TN / (TN + FP), 2)  # Specificity
                    positive_pred_value = round(TP / float(TP + FP), 2)  # PPV
                    negative_pred_value = round(TN / float(TN + FN), 2)  # NPV
                    auc = round(roc_auc_score(y, pred), 2)  # AUC
                    bacc = round(balanced_accuracy_score(y, pred), 2)  # balanced accuracy

                    # converting calculated metrics into a pandas dataframe to compare all models at the final
                    statistics = pd.DataFrame({'Bal-acc': bacc, "Sensitivity": sensitivity, "Specificity": specificity,
                                                "PPV": positive_pred_value,
                                                "NPV": negative_pred_value, 'Kappa': kappa, 'AUC': auc, 'MCC': mcc,
                                                'Accuracy': accuracy, }, index=[0])
                    return statistics

                ########################################################################################################################################

                # Create folds for cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=False, )

                # Run RF Model building - Bayesian hyperparameter search
                scorer = make_scorer(geometric_mean_score)

                opt_rf = BayesSearchCV(
                    RandomForestClassifier(),
                    hyperparameters,
                    n_iter=s_state['parameter_n_iter'],  # Number of parameter settings that are sampled
                    cv=cv,
                    scoring=scorer,
                    verbose=0,
                    refit=True,  # Refit the best estimator with the entire dataset.
                    random_state=s_state['parameter_random_state'],
                    n_jobs=s_state['parameter_n_jobs']
                )

                opt_rf.fit(x, y)

                st.write("Best parameters: %s" % opt_rf.best_params_)

                # k-fold cross-validation
                pred_rf, y_experimental, probs_classes, AD_fold, y_pred_ad, y_exp_ad = cros_val(x, y,
                                                                                                RandomForestClassifier(
                                                                                                    **opt_rf.best_params_),
                                                                                                cv)
                # Statistics k-fold cross-validation
                statistics = calc_statistics(y_experimental, pred_rf)
                # coverage
                coverage = round((len(y_exp_ad) / len(y_experimental)), 2)

                # converting calculated metrics into a pandas dataframe to save a xls
                model_type = "RF"

                result_type = "uncalibrated"

                metrics_rf_uncalibrated = statistics
                metrics_rf_uncalibrated['model'] = model_type
                metrics_rf_uncalibrated['result_type'] = result_type
                metrics_rf_uncalibrated['coverage'] = coverage

                st.header('**Metrics of uncalibrated model on the K-fold cross-validation**')

                # Bar chart Statistics k-fold cross-validation

                metrics_rf_uncalibrated_graph = metrics_rf_uncalibrated.filter(
                    items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC",
                            "coverage"])

                x_metrics = metrics_rf_uncalibrated_graph.columns
                y_metrics = metrics_rf_uncalibrated_graph.loc[0].values

                colors = ["#AEC6CF", "#B39EB5", "#FFB7CE", "#FFDAC1", "#FFD1DC", "#FDFD96", "#B5EAD7", "#CFCFC4", "#FFB347"]


                fig = go.Figure(data=[go.Bar(
                    x=x_metrics, y=y_metrics,
                    text=y_metrics,
                    textposition='auto',
                    marker_color=colors
                )])

                st.plotly_chart(fig)

                # Save the figure as a high-quality image
                img_buffer = io.BytesIO()
                fig.write_image(img_buffer, format="png", width=1200, height=800, scale=3)
                img_buffer.seek(0)

                # Add a download button
                st.download_button(
                    label="Download High-Quality Image",
                    data=img_buffer,
                    file_name="plot.png",
                    mime="image/png"
                )

                # (Continue with the rest of your modeling code, including calibration, external set evaluation, etc.)

                # Store variables in s_state
                s_state['opt_rf'] = opt_rf
                s_state['metrics_rf_uncalibrated'] = metrics_rf_uncalibrated
                # ... Store other results as needed

                ########################################################################################################################################
                # Download files
                ########################################################################################################################################

                st.header('**Download files**')
                        
                if selected_splitting == 'split_original' or selected_splitting == 'input_own':
                    frames = [metrics_rf_uncalibrated]

                else:
                    frames = [metrics_rf_uncalibrated]

            
                result = pd.concat(frames)

                result = result.round(2)

                # File download
                def filedownload(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
                    href = f'<a href="data:file/csv;base64,{b64}" download="metrics_rf.csv">Download CSV File - metrics</a>'
                    st.markdown(href, unsafe_allow_html=True)

                filedownload(result)

                def download_model(model):
                    output_model = pickle.dumps(model)
                    b64 = base64.b64encode(output_model).decode()
                    href = f'<a href="data:file/output_model;base64,{b64}" download= model_rf.pkl >Download generated model (PKL File)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                download_model(opt_rf)

            except Exception as e:
                st.error(f"An error occurred during modeling: {e}")
        else:
            st.write("Modeling has already been performed. Displaying results...")

    else:
        st.write("Click 'Run Modeling' to start the modeling process.")
    
    
