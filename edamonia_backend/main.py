import os
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import shutil

from edamonia_backend.logic.emb_models.emd import preprocess_and_generate_embeddings, load_embeddings
from edamonia_backend.logic.preprocessing.preprocess_data import process_data_embedded, preprocess_text_embedded
from edamonia_backend.logic.ranking_by_frequency.bm25lus import (
    reindex_bm25,
    load_bm25_index,
    ensure_unique_ids
)
from edamonia_backend.logic.responce_by_llm.llm import generate_assistant_response

app = FastAPI()

PRIMARY_CSV_DIR_PATH = "data/csv_files/primary_csv"
CLEANED_CSV_DIR_PATH = "data/csv_files/cleaned_csv"
INDEX_FILE_PATH = "data/csv_files/bm25_index.pkl"
COMBINED_FILE_CSV_PATH = "data/csv_files/combined_cleaned.csv"
EMBEDDINGS_FILE_PATH = "data/csv_files/embeddings.pkl"

os.makedirs(PRIMARY_CSV_DIR_PATH, exist_ok=True)
os.makedirs(CLEANED_CSV_DIR_PATH, exist_ok=True)

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

            # Генерація ембеддінгів
            preprocess_and_generate_embeddings(COMBINED_FILE_CSV_PATH, EMBEDDINGS_FILE_PATH)

            # Створення BM25 індексу
            reindex_bm25(COMBINED_FILE_CSV_PATH, INDEX_FILE_PATH)

            return {"message": f"Files processed and combined successfully: {', '.join(processed_files)}"}
        else:
            raise HTTPException(status_code=400, detail="No cleaned CSV files found to combine.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {str(e)}")


@app.post("/ask-bot/")
async def ask(query: QueryModel):
    """
    Повертає проранжовані документи на основі текстового запиту та генерує відповідь на основі цих документів.
    Використовує BM25 та ембеддінги для покращення результатів.
    """
    try:
        # Завантажуємо BM25 індекс
        bm25 = load_bm25_index(INDEX_FILE_PATH)

        if not bm25 or not hasattr(bm25, "get_scores"):
            raise HTTPException(status_code=500, detail="BM25 index is invalid or corrupted.")

        # Завантажуємо ембеддінги
        embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)
        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings are not available.")

        # Завантажуємо файл для отримання текстів документів
        df = pd.read_csv(COMBINED_FILE_CSV_PATH)
        if df.empty or "content" not in df.columns:
            raise HTTPException(status_code=500, detail="Combined file is missing or invalid.")

        # Попередня обробка запиту
        cleaned_query = preprocess_text_embedded(query.query)
        tokenized_query = cleaned_query.split()

        if not tokenized_query:
            raise HTTPException(status_code=400, detail="Query is empty or invalid after preprocessing.")

        # Ранжування документів за допомогою BM25
        scores_bm25 = bm25.get_scores(tokenized_query)
        top_k_bm25 = 3
        top_k_indices_bm25 = scores_bm25.argsort()[-top_k_bm25:][::-1]

        # Ранжування документів за допомогою ембеддінгів
        query_embedding = embeddings.embed_query(query.query)  # Припускаємо, що embed_query генерує ембеддинг для запиту
        document_embeddings = embeddings.embed_documents(df["content"].tolist())
        similarity_scores = embeddings.calculate_similarity(query_embedding, document_embeddings)
        top_k_embeddings_indices = similarity_scores.argsort()[-top_k_bm25:][::-1]

        # Формуємо результати з урахуванням BM25 та ембеддінгів
        ranked_documents = [
            {
                "id": df.iloc[idx]["news_id"],  # Вибір коректного ID
                "score": float(scores_bm25[idx] + similarity_scores[idx]),  # Можна комбінувати бали
                "content": df.iloc[idx]["content"]
            }
            for idx in top_k_bm25
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
