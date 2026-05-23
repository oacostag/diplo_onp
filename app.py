import streamlit as st
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from scraper import scrape, get_data, text_features
from utils import process_text_and_title
from predictor import Predictor

st.set_page_config(page_title="News Popularity Predictor", page_icon="📈", layout="centered")

@st.cache_resource
def load_predictor():
    return Predictor()

predictor = load_predictor()

st.title("📈 Predicción de Popularidad de Noticias")
st.markdown("Ingresa la URL de una noticia para predecir su rendimiento y clasificarla mediante Inteligencia Artificial.")

url_input = st.text_input("URL del artículo:", placeholder="https://mashable.com/article/example...")

if st.button("Analizar Artículo"):
    if url_input:
        with st.status("Analizando la URL...", expanded=True) as status:
            try:
                st.write("🌐 1. Haciendo web scraping...")
                url, html = scrape(url_input)
                soup = BeautifulSoup(html, "html.parser")
                
                title, n_subtitles, channel, n_images, n_videos, n_paragraphs, text = get_data(soup)
                
                st.write("⚙️ 2. Extrayendo características y análisis de sentimiento...")
                # Features for content
                content_features_df = text_features(url, text)
                if not content_features_df.empty:
                    content_compound_mean = content_features_df['compound'].mean()
                    content_n_words_mean = content_features_df['n_words'].mean()
                else:
                    content_compound_mean = 0.0
                    content_n_words_mean = 0.0
                    
                # Features for title
                title_features_df = text_features(url, title)
                if not title_features_df.empty:
                    title_compound_mean = title_features_df['compound'].mean()
                else:
                    title_compound_mean = 0.0
                    
                st.write("🧹 3. Limpiando y procesando el texto...")
                text_cleaned, title_cleaned, urls = process_text_and_title(text, title)
                
                n_urls = len(urls)
                len_title = len(title)
                n_words_title = len(title.split())
                avg_len_words_title = np.mean([len(x) for x in title.split()]) if n_words_title > 0 else 0
                
                len_content = len(text)
                n_words_content = len(text.split())
                avg_len_words_content = np.mean([len(x) for x in text.split()]) if n_words_content > 0 else 0

                features_dict = {
                    "n_images": n_images,
                    "n_videos": n_videos,
                    "title_compound_mean": title_compound_mean,
                    "content_compound_mean": content_compound_mean,
                    "content_n_words_mean": content_n_words_mean,
                    "n_urls": n_urls,
                    "len_title": len_title,
                    "n_words_title": n_words_title,
                    "avg_len_words_title": avg_len_words_title,
                    "len_content": len_content,
                    "n_words_content": n_words_content,
                    "avg_len_words_content": avg_len_words_content
                }

                st.write("🧠 4. Realizando inferencia con los modelos de Deep Learning...")
                shares_pred, cluster_pred = predictor.predict(title_cleaned, text_cleaned, features_dict)
                
                status.update(label="¡Análisis completado!", state="complete", expanded=False)
                
                st.success("¡Predicción exitosa!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Visitas/Shares Estimadas", value=f"{int(shares_pred):,}")
                    
                with col2:
                    st.metric(label="Clúster Asignado (Segmento)", value=f"Clúster {cluster_pred}")
                    
                with st.expander("Ver detalles extraídos del artículo"):
                    st.markdown(f"**Título:** {title}")
                    st.markdown(f"**Imágenes:** {n_images} | **Videos:** {n_videos} | **Párrafos:** {n_paragraphs}")
                    st.markdown(f"**Sentimiento Promedio del Contenido:** {content_compound_mean:.2f}")

            except Exception as e:
                status.update(label="Error en el análisis", state="error", expanded=True)
                st.error(f"Se produjo un error al procesar la URL: {str(e)}")
    else:
        st.warning("Por favor, ingresa una URL válida.")
