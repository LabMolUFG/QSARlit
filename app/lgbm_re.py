import streamlit as st
from st_aggrid import AgGrid

import base64
from io import BytesIO
import warnings
import io
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os

import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import pickle

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
        s_state['parameter_random_state'] = st.sidebar.number_input(
            'Seed number (random_state)',
            value=s_state['parameter_random_state'],
            step=1,
            key='random_state'
        )

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
        st.write(df)

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
        columns_to_delete = st.multiselect(
            'Select columns to delete',
            columns,
            default=default_columns_to_delete,
            key='columns_to_delete_widget'
        )

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
    # Select target column
    ########################################################################################################################################
    with st.sidebar.header('2. Select target column'):
        # Ensure default value is valid
        if 'name_activity' in s_state and s_state['name_activity'] in df.columns:
            index = df.columns.tolist().index(s_state['name_activity'])
        else:
            index = 0  # or any other default index
            s_state['name_activity'] = df.columns[index]

        name_activity = st.sidebar.selectbox(
            'Select the target column (continuous variable for regression)',
            df.columns,
            index=index,
            key='name_activity'
        )

        if len(name_activity) > 0:
            if name_activity not in df.columns:
                st.error(f"The column '{name_activity}' is not in the dataframe.")
    st.sidebar.write('---')

    ########################################################################################################################################
    # Data splitting
    ########################################################################################################################################

    # Selecting X and y from input file
    x = df.drop(columns=[name_activity]).values  # All columns except the target column
    y = df[name_activity].values  # The target column

    ########################################################################################################################################
    # Sidebar - Specify parameter settings
    ########################################################################################################################################

    st.sidebar.header('5. Set Parameters - Bayesian hyperparameter search')

    # Choose the general hyperparameters
    st.sidebar.subheader('General Parameters')

    s_state['parameter_n_iter'] = st.sidebar.slider(
        'Number of iterations (n_iter)',
        1,
        1000,
        s_state['parameter_n_iter'],
        1,
        key='n_iter'
    )

    st.sidebar.write('---')
    s_state['parameter_n_jobs'] = st.sidebar.selectbox(
        'Number of jobs to run in parallel (n_jobs)',
        options=[-1, 1],
        index=0,
        key='n_jobs'
    )

    # Choose the hyperparameters intervals to be tested
    st.sidebar.subheader('Learning Hyperparameters')

    hyperparameters = {
        'num_leaves': (20, 50),
        'max_depth': (5, 15),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'n_estimators': (100, 1000),
        'min_child_samples': (5, 30),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
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

                def calc_statistics(y_true, y_pred):
                    """
                    Calculate and return regression statistics.

                    Parameters:
                    y_true (array-like): True target values.
                    y_pred (array-like): Predicted target values.

                    Returns:
                    pd.DataFrame: DataFrame containing regression metrics.
                    """
                    # Calculate regression metrics
                    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
                    mse = mean_squared_error(y_true, y_pred)    # Mean Squared Error
                    rmse = np.sqrt(mse)                         # Root Mean Squared Error
                    r2 = r2_score(y_true, y_pred)               # R-squared

                    # Initialize the DataFrame
                    statistics = {
                        'MAE': [mae],
                        'MSE': [mse],
                        'RMSE': [rmse],
                        'R²': [r2]
                    }
                    statistics_df = pd.DataFrame(statistics)
                    return statistics_df

                ########################################################################################################################################

                # Create folds for cross-validation
                cv = KFold(n_splits=5, shuffle=True, random_state=s_state['parameter_random_state'])

                # Run LightGBM Regression with Bayesian hyperparameter search
                opt_lgbm = BayesSearchCV(
                    lgb.LGBMRegressor(random_state=s_state['parameter_random_state']),
                    hyperparameters,
                    n_iter=s_state['parameter_n_iter'],  # Number of parameter settings that are sampled
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    verbose=0,
                    refit=True,  # Refit the best estimator with the entire dataset.
                    random_state=s_state['parameter_random_state'],
                    n_jobs=s_state['parameter_n_jobs']
                )

                opt_lgbm.fit(x, y)

                st.write("Best parameters: %s" % opt_lgbm.best_params_)

                # Cross-validation predictions
                from sklearn.model_selection import cross_val_predict

                y_pred_cv = cross_val_predict(
                    opt_lgbm.best_estimator_,
                    x,
                    y,
                    cv=cv,
                    n_jobs=s_state['parameter_n_jobs']
                )

                # Statistics k-fold cross-validation
                statistics = calc_statistics(y, y_pred_cv)

                # Display metrics
                st.header('**Metrics on K-Fold Cross-Validation**')
                st.write(statistics)

                # Plotting the metrics
                metrics_to_plot = ['MAE', 'MSE', 'RMSE', 'R²']
                x_metrics = metrics_to_plot
                y_metrics = statistics.loc[0, metrics_to_plot].values

                colors = ["#AEC6CF", "#B39EB5", "#FFB7CE", "#FFDAC1"]

                fig = go.Figure(data=[go.Bar(
                    x=x_metrics,
                    y=y_metrics,
                    text=[f"{val:.2f}" for val in y_metrics],
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

                # Store variables in s_state
                s_state['opt_lgbm'] = opt_lgbm
                s_state['statistics'] = statistics

                ########################################################################################################################################
                # Download files
                ########################################################################################################################################

                st.header('**Download files**')

                # File download
                def filedownload(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
                    href = f'<a href="data:file/csv;base64,{b64}" download="metrics_lgbm.csv">Download CSV File - Metrics</a>'
                    st.markdown(href, unsafe_allow_html=True)

                filedownload(statistics)

                def download_model(model):
                    output_model = pickle.dumps(model)
                    b64 = base64.b64encode(output_model).decode()
                    href = f'<a href="data:file/output_model;base64,{b64}" download="model_lgbm.pkl">Download Trained Model (PKL File)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                download_model(opt_lgbm.best_estimator_)

            except Exception as e:
                st.error(f"An error occurred during modeling: {e}")
        else:
            st.write("Modeling has already been performed. Displaying results...")

    else:
        st.write("Click 'Run Modeling' to start the modeling process.")
