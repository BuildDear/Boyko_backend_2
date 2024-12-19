import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from edamonia_backend.logic.preprocessing.preprocess_data import preprocess_text_embedded
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Hugging Face token is not set in the .env file.")

login(hf_token)

# Initialize the model
model = SentenceTransformer("intfloat/e5-small")

def preprocess_and_generate_embeddings(input_path, output_path):
    """
    Обробляє текстові дані з CSV файлу, генерує ембеддинги для кожного документу та зберігає їх.
    :param input_path: Шлях до вхідного CSV, що містить колонки 'content' та 'news_id'
    :param output_path: Шлях до файлу, де будуть збережені ембеддинги
    :return: None
    """
    try:
        documents_df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {input_path} не знайдено.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Файл {input_path} порожній.")

    if "content" not in documents_df.columns:
        raise ValueError("Missing 'content' column in input CSV.")

    corpus = documents_df['content'].dropna().tolist()
    processed_corpus = [preprocess_text_embedded(text) for text in tqdm(corpus, total=len(corpus))]
    embeddings = generate_embeddings(processed_corpus)
    save_embeddings(output_path, embeddings)
    print(f"Ембеддинги збережено у {output_path}.")

def generate_embeddings(texts):
    """
    Генерує ембеддинги для списку текстів за допомогою SentenceTransformer.
    :param texts: Список текстів для генерації ембеддингів
    :return: numpy масив ембеддингів
    """
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    return np.array(embeddings)

def save_embeddings(output_path, embeddings):
    """
    Зберігає ембеддинги в файл
    :param output_path: шлях до файлу
    :param embeddings: ембеддинги для збереження
    """
    np.save(output_path, embeddings)

def load_embeddings(input_path):
    """
    Завантажує ембеддинги з файлу
    :param input_path: шлях до файлу з ембеддингами
    :return: ембеддинги
    """
    return np.load(input_path)