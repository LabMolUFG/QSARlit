import streamlit as st
from multiapp import MultiApp
import home, cur, desc, rf, svm, lgbm, vs, rf_re, svm_re, lgbm_re
import utils
import pandas as pd

# Instantiate the MultiApp object
app = MultiApp()

# Add all your applications here
app.add_app("Home", home.app)
app.add_app("Curation for modeling", cur.app)
app.add_app("Calculate Descriptors", desc.app)
app.add_app("Random Forest - Classification", rf.app)
app.add_app("Support Vector Classification", svm.app)
app.add_app("LightGBM - Classification", lgbm.app)
app.add_app("Random Forest - Regressor", rf_re.app)
app.add_app("Support Vector Regressor", svm_re.app)
app.add_app("LightGBM - Regressor", lgbm_re.app)
app.add_app("Virtual Screening", vs.app)

# Instantiate the Custom_Components class
cc = utils.Custom_Components()

# The main app
s_state = st.session_state

def run_app():
    # Initialize df in session state if not already present
    if 'df' not in s_state:
        s_state.df = None

    # Run the multiapp
    app.run(s_state.df, s_state)

# Run the app
run_app()
