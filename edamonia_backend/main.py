import os
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import shutil

from edamonia_backend.logic.preprocessing.preprocess_data import process_data_embedded, preprocess_text_embedded
from edamonia_backend.logic.ranking_by_frequency.bm25lus import (
    reindex_bm25,
    load_bm25_index,
    ensure_unique_ids
)
from edamonia_backend.logic.ranking_by_frequency.tf_idf import load_tfidf_index, get_tfidf_scores, reindex_tfidf
from edamonia_backend.logic.responce_by_llm.llm import generate_assistant_response

app = FastAPI()

PRIMARY_CSV_DIR_PATH = "data/csv_files/primary_csv"
CLEANED_CSV_DIR_PATH = "data/csv_files/cleaned_csv"
BM25PUS_INDEX_FILE_PATH = "data/csv_files/bm25_index.pkl"
TFIDF_INDEX_FILE_PATH = "data/csv_files/tfidf_index.pkl"
COMBINED_FILE_CSV_PATH = "data/csv_files/combined_cleaned.csv"


os.makedirs(PRIMARY_CSV_DIR_PATH, exist_ok=True)
os.makedirs(CLEANED_CSV_DIR_PATH, exist_ok=True)

class QueryRequest(BaseModel):
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
    Обробляє завантажені файли CSV, об'єднує їх в один файл, забезпечує унікальність ID, і створює BM25 індекс.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        processed_files = []

        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file name found in uploaded file")

            file_name, file_ext = os.path.splitext(file.filename)

            if file_ext.lower() == ".csv":
                primary_file_name = f"{file_name}_primary{file_ext}"
                input_file_path = os.path.join(PRIMARY_CSV_DIR_PATH, primary_file_name)

                # Зберігаємо в primary
                with open(input_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Обробляємо файл
                cleaned_file_name = f"{file_name}_cleaned{file_ext}"
                output_file_path = os.path.join(CLEANED_CSV_DIR_PATH, cleaned_file_name)
                process_data_embedded(input_file_path, output_file_path)
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

            # Створення BM25 індексу
            reindex_bm25(COMBINED_FILE_CSV_PATH, BM25PUS_INDEX_FILE_PATH)
            reindex_tfidf(COMBINED_FILE_CSV_PATH, TFIDF_INDEX_FILE_PATH)

            return {"message": f"Files processed and combined successfully: {', '.join(processed_files)}"}
        else:
            raise HTTPException(status_code=400, detail="No cleaned CSV files found to combine.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {str(e)}")


def weighted_voting(bm25_scores, tfidf_scores, alpha=0.5):
    return alpha * bm25_scores + (1 - alpha) * tfidf_scores


@app.post("/ask-bot/")
async def ask(query: QueryRequest):
    try:

        bm25 = load_bm25_index(BM25PUS_INDEX_FILE_PATH)
        vectorizer, tfidf_matrix = load_tfidf_index(TFIDF_INDEX_FILE_PATH)

        if not bm25 or not hasattr(bm25, "get_scores"):
            raise HTTPException(status_code=500, detail="BM25 index is invalid or corrupted.")

        df = pd.read_csv(COMBINED_FILE_CSV_PATH)
        if df.empty or "content" not in df.columns:
            raise HTTPException(status_code=500, detail="Combined file is missing or invalid.")

        # Попередня обробка запиту
        cleaned_query = preprocess_text_embedded(query.query)
        tokenized_query = cleaned_query.split()

        if not tokenized_query:
            raise HTTPException(status_code=400, detail="Query is empty or invalid after preprocessing.")

        # Отримуємо оцінки з BM25
        bm25_scores = bm25.get_scores(tokenized_query)

        # Отримуємо оцінки з TF-IDF
        tfidf_scores = get_tfidf_scores(query.query, vectorizer, tfidf_matrix)

        # Зважене голосування
        combined_scores = weighted_voting(bm25_scores, tfidf_scores, alpha=0.5)

        # Формуємо результати
        top_k = 3
        top_k_indices = combined_scores.argsort()[-top_k:][::-1]

        ranked_documents = [
        {
            "id": df.iloc[idx]["news_id"],
            "score": float(combined_scores[idx]),
            "content": df.iloc[idx]["content"]
        }
        for idx in top_k_indices
    ]

        # Формуємо контекст із ранжованих документів
        context = ' '.join(doc['content'] for doc in ranked_documents)

        # Генеруємо відповідь асистента
        assistant_response = generate_assistant_response(query.query + " " + context)

        return {
            "query": query.query,
            "ranked_documents": ranked_documents,
            "assistant_response": assistant_response
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="BM25 index file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")

'''
Useful commands

poetry remove lib
poetry add lib
poetry install
poetry shell
uvicorn edamonia_backend.main:app --reload 
poetry run python data/datasets/dataset.py
'''