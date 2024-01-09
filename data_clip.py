import pandas as pd
import matplotlib.pyplot as plt
from data_load import load_data
import glob
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
def calculate_similarity(text, image_data, model_id="clip-ViT-B-32-multilingual-v1", top_k=5):
    # Utilisation du modèle SentenceTransformers
    model = SentenceTransformer(model_id)
    
    # Encodage du texte
    text_embedding = model.encode([text], convert_to_tensor=True)

    # Encodage des images dans la base de données
    image_embeddings = model.encode([str(filepath) for filepath in image_data], batch_size=500, convert_to_tensor=True, show_progress_bar=True)
    #image_embeddings=np.load("precomputed_image_embeddings.npy")
    # Calcul de la similarité
    #scores = util.pytorch_cos_sim(text_embedding, image_embeddings)[0].cpu().detach().numpy()
    scores = util.pytorch_cos_sim(text_embedding,  image_embeddings).cpu().detach().numpy().flatten()
    # Récupération des indices des images les plus similaires
    top_indices = np.argsort(-scores)[:top_k]
    
    #Tri des indices en fonction des scores de similarité
    sorted_indices = sorted(top_indices, key=lambda x: scores[x], reverse=True)
    # Affichage des résultats
    matched_images = [Image.open(image_data[idx]) for idx in  top_indices]
    
    # Extraction des noms de fichiers correspondants
    matched_image_names = [image_data[idx] for idx in sorted_indices]
    return  sorted_indices, scores, matched_images, matched_image_names
def clip(text, image_data, model_id="clip-ViT-B-32-multilingual-v1", top_k=5):
    # Utilisation du modèle SentenceTransformers
    model = SentenceTransformer(model_id)
    
    # Encodage du texte
    text_embedding = model.encode([text], convert_to_tensor=True)

    # Encodage des images dans la base de données
    #image_embeddings = model.encode([str(filepath) for filepath in image_data], batch_size=500, convert_to_tensor=True, show_progress_bar=True)
    image_embeddings=np.load("precomputed_image_embeddings.npy")
    # Calcul de la similarité
    #scores = util.pytorch_cos_sim(text_embedding, image_embeddings)[0].cpu().detach().numpy()
    scores = util.pytorch_cos_sim(text_embedding,  image_embeddings).cpu().detach().numpy().flatten()
    # Récupération des indices des images les plus similaires
    top_indices = np.argsort(-scores)[:top_k]
    
    #Tri des indices en fonction des scores de similarité
    sorted_indices = sorted(top_indices, key=lambda x: scores[x], reverse=True)
    # Affichage des résultats
    matched_images = [Image.open(image_data[idx]) for idx in  top_indices]
    
    # Extraction des noms de fichiers correspondants
    matched_image_names = [image_data[idx] for idx in sorted_indices]
    
    return  sorted_indices, scores, matched_images, matched_image_names
def clip_image_text(reference_image_path, image_data, model_id="clip-ViT-B-32-multilingual-v1", top_k=5):
    # Utilisation du modèle SentenceTransformers
    model = SentenceTransformer(model_id)

    # Encodage de l'image de référence
    reference_image_embedding = model.encode([str(reference_image_path)], convert_to_tensor=True)

    # Encodage des images dans la base de données
    image_embeddings=np.load("precomputed_image_embeddings.npy")

    # Calcul de la similarité entre l'image de référence et les autres images
    scores = util.pytorch_cos_sim(reference_image_embedding, image_embeddings).cpu().detach().numpy().flatten()

    # Récupération des indices des images les plus similaires
    top_indices = np.argsort(-scores)[:top_k]

    # Tri des indices en fonction des scores de similarité
    sorted_indices = sorted(top_indices, key=lambda x: scores[x], reverse=True)

    # Affichage des résultats
    matched_images = [Image.open(image_data[idx]) for idx in top_indices]

    # Extraction des noms de fichiers correspondants
    matched_image_names = [image_data[idx] for idx in top_indices]
    
    return sorted_indices, scores, matched_images, matched_image_names
