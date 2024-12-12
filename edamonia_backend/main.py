import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import shutil

from sentence_transformers import SentenceTransformer

from edamonia_backend.logic.emb_models.emd import preprocess_and_generate_embeddings, load_embeddings
from edamonia_backend.logic.preprocessing.chunking import process_txt, save_chunks_to_csv, process_pdf, process_docx, \
    process_json
from edamonia_backend.logic.preprocessing.preprocess_data import process_data_embedded, preprocess_text_embedded
from edamonia_backend.logic.ranking_by_frequency.bm25lus import (
    reindex_bm25,
    load_bm25_index,
    ensure_unique_ids
)
from edamonia_backend.logic.ranking_by_frequency.tf_idf import load_tfidf_index, get_tfidf_scores, reindex_tfidf
from edamonia_backend.logic.responce_by_llm.llm import generate_assistant_response

app = FastAPI()

model = SentenceTransformer("intfloat/e5-small")

PRIMARY_CSV_DIR_PATH = "data/csv_files/primary_csv"
CLEANED_CSV_DIR_PATH = "data/csv_files/cleaned_csv"
BM25PUS_INDEX_FILE_PATH = "data/csv_files/bm25_index.pkl"
TFIDF_INDEX_FILE_PATH = "data/csv_files/tfidf_index.pkl"
COMBINED_FILE_CSV_PATH = "data/csv_files/combined_cleaned.csv"
EMBEDDINGS_FILE_PATH = "data/csv_files/combined_cleaned_embeddings.npy"


os.makedirs(PRIMARY_CSV_DIR_PATH, exist_ok=True)
os.makedirs(CLEANED_CSV_DIR_PATH, exist_ok=True)

def weighted_voting(bm25_scores, tfidf_scores, embedding_scores, alpha=0.4, beta=0.3, gamma=0.3):
    return alpha * bm25_scores + beta * tfidf_scores + gamma * embedding_scores

class QueryModel(BaseModel):
    query: str


@app.get("/main", summary="Check API status", response_description="API is running.")
async def main():
    """
    Перевіряє, чи працює сервер.
    """
    return {"message": "Files uploaded successfully"}


@app.post("/process-file/", summary="Process and combine uploaded CSV files")
async def process_csv(files: List[UploadFile] = File(...)):
    """
    Обробляє завантажені файли CSV, об'єднує їх в один файл, забезпечує унікальність ID, створює BM25 індекс та генерує ембеддінги.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        processed_files = []

        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file name found in uploaded file")

            file_name, file_ext = os.path.splitext(file.filename)
            input_file_path = os.path.join(PRIMARY_CSV_DIR_PATH, file.filename)

            with open(input_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            if file_ext.lower() == ".csv":
                cleaned_file_name = f"{file_name}_cleaned.csv"
                output_file_path = os.path.join(CLEANED_CSV_DIR_PATH, cleaned_file_name)
                process_data_embedded(input_file_path, output_file_path)
                processed_files.append(file.filename)

            elif file_ext.lower() == ".txt":
                chunks = process_txt(input_file_path)
                output_file_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_path)
                processed_files.append(file.filename)

            elif file_ext.lower() == ".pdf":
                chunks = process_pdf(input_file_path)
                output_file_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_path)
                processed_files.append(file.filename)

            elif file_ext.lower() == ".docx":
                chunks = process_docx(input_file_path)
                output_file_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_path)
                processed_files.append(file.filename)

            elif file_ext.lower() == ".json":
                chunks = process_json(input_file_path)
                output_file_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_path)
                processed_files.append(file.filename)

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        # Об'єднання очищених файлів
        cleaned_files = [f for f in os.listdir(CLEANED_CSV_DIR_PATH) if f.endswith(".csv")]
        dfs = [pd.read_csv(os.path.join(CLEANED_CSV_DIR_PATH, cf)) for cf in cleaned_files]

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = ensure_unique_ids(combined_df, "news_id")
            combined_df.to_csv(COMBINED_FILE_CSV_PATH, index=False)

            # Генерація ембеддінгів
            preprocess_and_generate_embeddings(COMBINED_FILE_CSV_PATH)

            # Створення BM25 індексу
            reindex_bm25(COMBINED_FILE_CSV_PATH, BM25PUS_INDEX_FILE_PATH)
            reindex_tfidf(COMBINED_FILE_CSV_PATH, TFIDF_INDEX_FILE_PATH)

            return {"message": f"Files processed and combined successfully: {', '.join(processed_files)}"}
        else:
            raise HTTPException(status_code=400, detail="No cleaned CSV files found to combine.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {str(e)}")



@app.post("/ask-bot/")
async def ask(query: QueryModel):
    try:
        # Перевірка наявності необхідних файлів
        if not os.path.exists(BM25PUS_INDEX_FILE_PATH):
            raise HTTPException(status_code=404, detail="BM25 index file not found.")
        if not os.path.exists(TFIDF_INDEX_FILE_PATH):
            raise HTTPException(status_code=404, detail="TF-IDF index file not found.")
        if not os.path.exists(EMBEDDINGS_FILE_PATH):
            raise HTTPException(status_code=404, detail="Embeddings file not found.")
        if not os.path.exists(COMBINED_FILE_CSV_PATH):
            raise HTTPException(status_code=404, detail="Combined CSV file not found.")

        # Завантаження BM25 індексу
        bm25 = load_bm25_index(BM25PUS_INDEX_FILE_PATH)
        if not bm25 or not hasattr(bm25, "get_scores"):
            raise HTTPException(status_code=500, detail="BM25 index is invalid or corrupted.")

        # Завантаження TF-IDF індексу
        vectorizer, tfidf_matrix = load_tfidf_index(TFIDF_INDEX_FILE_PATH)

        # Завантаження ембеддингів
        document_embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)
        if document_embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings are not available.")

        # Завантаження даних документів
        df = pd.read_csv(COMBINED_FILE_CSV_PATH)
        if df.empty or "content" not in df.columns:
            raise HTTPException(status_code=500, detail="Combined file is missing or invalid.")

        # Попередня обробка запиту
        cleaned_query = preprocess_text_embedded(query.query)
        tokenized_query = cleaned_query.split()

        if not tokenized_query:
            raise HTTPException(status_code=400, detail="Query is empty or invalid after preprocessing.")

        # Отримання оцінок з BM25
        bm25_scores = bm25.get_scores(tokenized_query)

        # Отримання оцінок з TF-IDF
        tfidf_scores = get_tfidf_scores(query.query, vectorizer, tfidf_matrix)

        # Генерація ембеддингу для запиту
        query_embedding = model.encode([query.query], show_progress_bar=False)[0]

        # Обчислення косинусної схожості
        similarity_scores = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Об'єднання оцінок зваженим голосуванням
        combined_scores = weighted_voting(bm25_scores, tfidf_scores, similarity_scores, alpha=0.4, beta=0.3, gamma=0.3)

        # Вибір топ-K документів
        top_k = 3
        top_k_indices = combined_scores.argsort()[-top_k:][::-1]

        # Формування списку результатів
        ranked_documents = [
            {
                "id": df.iloc[idx]["news_id"],
                "score": float(combined_scores[idx]),
                "content": df.iloc[idx]["content"]
            }
            for idx in top_k_indices
        ]

        # Формування контексту
        context = ' '.join(doc['content'] for doc in ranked_documents)

        # Генерація відповіді асистента
        assistant_response = generate_assistant_response(query.query + " " + context)

        return {
            "query": query.query,
            "ranked_documents": ranked_documents,
            "assistant_response": assistant_response
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")