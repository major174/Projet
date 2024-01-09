import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import base64
from PIL import Image
from io import BytesIO
from data_load import load_data
from data_clip import calculate_similarity, clip, clip_image_text
from data_clean import cleaning
from data_link_image import link_image
from util import config

st.set_page_config(
    layout="wide"
)
config()
st.markdown("<h1 style='text-align: center;font-family: \"Times New Roman\", Times, serif; font-size: 60px; color : white;background-color: orange;'>Competitor Analysis</h1>", unsafe_allow_html=True)

path = "data/data_csv/data_vrai.csv"
df_text, imagePatche = load_data(path)


def path_to_image_html(path):
    return f'<img src="data:image/jpeg;base64,{path}" width="300", height="300" >'


def image_to_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image


def is_image_empty(image):
    # Check if all pixel values are zero (i.e., the image is empty)
    return np.all(image == 0)

# Définition de la taille de la police pour les éléments spécifiques
st.markdown(
    """
    <style>
        /* Titre de la section */
        .streamlit-section st-text-area label {
            font-size: 3rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.form("my_form"):
    st.markdown("<h2 style='font-family: \"Times New Roman\", Times, serif; font-size: 30px;'>Product Details</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
       
        product_line = st.selectbox("Catégorie:", ['beaute-hygiene-sante', 'electronique', 'ordinateurs-accessoires-informatique', 'telephone_tablette'])
          # Utilisez la balise HTML pour le label de la catégorie avec une taille de police personnalisée
        
    with col2:
        product_id = st.text_input("Modele")
    with col3:
        
        product_title = st.text_input("Nom du produit")
    with col4:
        product_description = st.text_area("Description")
    with col5:
        uploaded_file = st.file_uploader(label="Product image", type=['jpg', 'png', 'jpeg'])

        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}

            # Chargez l'image à partir du fichier téléchargé
            user_product_image = load_image(uploaded_file)

            # Vérifiez si l'image est vide
            if is_image_empty(user_product_image):
                st.warning("L'image téléchargée est vide.")
            else:
                st.image(user_product_image, width=300)
        else:
            st.warning("Veuillez télécharger une image.")
            user_product_image = None
    

    submitted = st.form_submit_button("Search competitors",type='primary')

    if submitted:

        if user_product_image is not None and not is_image_empty(user_product_image):
            sorted_indices, scores, matched_images, matched_image_names = clip_image_text(user_product_image, imagePatche, top_k=5)
        else:
            prompts = pd.DataFrame({'prompt': [product_line + product_id + product_title + product_description]})
            prompt = cleaning(prompts, 'prompt')
            sorted_indices, scores, matched_images, matched_image_names = clip(prompt['prompt'][0], imagePatche, top_k=10)
        
        numbers = link_image(matched_image_names)
     
        df_list = []
        image_base64_list = []

        for i in range(len(numbers)):
            df_list.append({
                'Image': matched_images[i],
                'Nom_du_produit': df_text['Nom'][numbers[i]],
                'Prix(FCFA)': df_text['prix'][numbers[i]],
                'Score': scores[i],
            })
            image_base64_list.append(image_to_base64(matched_images[i]))

        df = pd.DataFrame(df_list)
        df['Image'] = image_base64_list

        # Réorganiser les colonnes dans l'ordre souhaité
        df = df[['Image', 'Nom_du_produit', 'Prix(FCFA)', 'Score']]
        
        styled_html = (
        df.to_html(escape=False, col_space='250px', formatters=dict(Image=path_to_image_html), index=False)
        .replace('<table border="1" class="dataframe">', '<table style="border-collapse: collapse; width: 100%; font-family: \'Times New Roman\', Times, serif;font-size: 25px;color: white;background-color: black;">')
        .replace('<thead>', '<thead style="font-size: 25px; color: white;">')
    )

    # Affichez la chaîne HTML stylisée
        st.markdown(styled_html, unsafe_allow_html=True)