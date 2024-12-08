import os
import pickle

import pandas as pd
from rank_bm25 import BM25Plus

def save_bm25_index(bm25: BM25Plus, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(bm25, f)


def load_bm25_index(file_path: str) -> BM25Plus:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_cleaned_files(directory: str):
    documents = []

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    documents.append(content)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

    return documents


def reindex_bm25(cleaned_file: str, index_file: str):
    try:
        df = pd.read_csv(cleaned_file)
    except FileNotFoundError:
        raise ValueError(f"Cleaned file not found: {cleaned_file}")
    except Exception as e:
        raise ValueError(f"Error reading cleaned file: {str(e)}")

    # Перевіряємо наявність необхідної колонки, наприклад "content"
    if "content" not in df.columns:
        raise ValueError(f"The required column 'content' is missing in the cleaned file: {cleaned_file}")

    # Формуємо корпус документів
    documents = df["content"].dropna().tolist()  # Вилучаємо порожні значення
    if not documents:
        raise ValueError("No valid content found in the cleaned file.")

    # Токенізуємо корпус документів
    tokenized_corpus = [doc.split() for doc in documents]

    # Створюємо BM25 індекс
    bm25 = BM25Plus(tokenized_corpus)

    # Зберігаємо індекс у файл
    save_bm25_index(bm25, index_file)
    print(f"BM25Plus index updated and saved to {index_file}")



def ensure_unique_ids(dataframe, id_column):
    id_counts = dataframe[id_column].value_counts()
    duplicates = id_counts[id_counts > 1].index

    for duplicate_id in duplicates:
        duplicate_indices = dataframe[dataframe[id_column] == duplicate_id].index
        for i, idx in enumerate(duplicate_indices):
            if i == 0:
                continue  # Перший запис залишаємо без змін
            dataframe.at[idx, id_column] = f"{dataframe.at[idx, id_column]}_{i}"

    return dataframe