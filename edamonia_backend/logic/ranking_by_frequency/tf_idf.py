from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_tfidf_scores(query: str, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    return cosine_similarities


def save_tfidf_index(vectorizer, tfidf_matrix, index_file: str):
    with open(index_file, 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix), f)
    print(f"TF-IDF index saved to {index_file}")


def load_tfidf_index(tfidf_index_file: str):
    try:
        with open(tfidf_index_file, 'rb') as f:
            vectorizer, tfidf_matrix = pickle.load(f)
        print(f"TF-IDF index loaded from {tfidf_index_file}")
        return vectorizer, tfidf_matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"TF-IDF index file {tfidf_index_file} not found.")
    except Exception as e:
        raise ValueError(f"Error loading TF-IDF index: {str(e)}")


def reindex_tfidf(cleaned_file: str, index_file: str):
    try:
        # Читаємо очищений CSV файл
        df = pd.read_csv(cleaned_file)
    except FileNotFoundError:
        raise ValueError(f"Cleaned file not found: {cleaned_file}")
    except Exception as e:
        raise ValueError(f"Error reading cleaned file: {str(e)}")

    # Перевірка наявності колонки "content"
    if "content" not in df.columns:
        raise ValueError(f"The required column 'content' is missing in the cleaned file: {cleaned_file}")

    # Формуємо корпус документів
    documents = df["content"].dropna().tolist()  # Вилучаємо порожні значення
    if not documents:
        raise ValueError("No valid content found in the cleaned file.")

    # Створюємо TF-IDF векторизатор
    vectorizer = TfidfVectorizer()

    # Тренуємо векторизатор і отримуємо TF-IDF матрицю
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Зберігаємо індекс у файл
    save_tfidf_index(vectorizer, tfidf_matrix, index_file)
