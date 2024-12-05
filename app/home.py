import os
import streamlit as st
import base64  # Add this line for the base64 module
from PIL import Image

def app(_, s_state):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the image file
    image_path = os.path.join(current_dir, 'qsartlit-logo.png')

    # Center-align the image using CSS
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{image_to_base64(image_path)}" alt="Logo" width="50%">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Center-align the markdown content
    st.markdown("""
    <style>
    .centered-text {
        text-align: center;
        margin-top: 2%;
    }
    </style>
    <p class="centered-text">
    This app allows you to curate data, calculate molecular descriptors, develop Machine Learning models, and perform virtual screening 
    for computational toxicology and drug discovery projects.
    </p>
    <div class="centered-text">
    <strong>Credits</strong><br>
    - App built in Python + Streamlit by Igor Henrique Sanches (<a href="http://lattes.cnpq.br/8640209397163468" target="_blank">Lattes</a>, <a href="https://www.linkedin.com/in/sanches-igor" target="_blank">LinkedIn</a>), Joyce Vila Verde Bastos Borba, Sabrina Silva-Mendonça, Jade Milhomem Lemos (<a href="http://lattes.cnpq.br/0140760039980984" target="_blank">Lattes</a>, <a href="https://www.linkedin.com/in/jade-milhomem-lemos-7119aa129/" target="_blank">LinkedIn</a>), Francisco L. Feitosa (<a href="https://orcid.org/0009-0006-0619-4514" target="_blank">Orcid</a>, <a href="https://www.linkedin.com/in/franciscofeitosafl/" target="_blank">LinkedIn</a>), Ester Sousa, José Teófilo Moreira Filho (<a href="http://lattes.cnpq.br/3464351249761623" target="_blank">Lattes</a>, <a href="https://scholar.google.com/citations?user=0I1GiOsAAAAJ&hl=pt-BR" target="_blank">Google Scholar</a>, <a href="https://orcid.org/0000-0002-0777-280X" target="_blank">ORCID</a>), Henric Gil, Bruno J. Neves, Rodolpho C. Braga and Carolina Horta Andrade.
    """, unsafe_allow_html=True)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if __name__ == "__main__":
    app(None, None)
