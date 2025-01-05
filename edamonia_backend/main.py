import os
from typing import List
import importlib
from data.datasets.gen_test_dataset import generate_10_data
from pydantic import Field, field_validator
from datetime import datetime
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from pydantic import BaseModel
import shutil

from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

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
from edamonia_backend.logic.responce_by_llm.llm import generate_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Дозволити всі методи (GET, POST тощо)
    allow_headers=["*"],  # Дозволити всі заголовки
)

model = SentenceTransformer("intfloat/e5-small")

PRIMARY_CSV_DIR_PATH = "data/csv_files/primary_csv"
PRIMARY_DIR_PATH = "data/primary"
CLEANED_CSV_DIR_PATH = "data/csv_files/cleaned_csv"
BM25PUS_INDEX_FILE_PATH = "data/csv_files/bm25_index.pkl"
TFIDF_INDEX_FILE_PATH = "data/csv_files/tfidf_index.pkl"
COMBINED_FILE_CSV_PATH = "data/csv_files/combined/combined_cleaned.csv"
PRIMARY_COMBINED_FILE_CSV_PATH = "data/csv_files/combined/combined_primary.csv"
EMBEDDINGS_FILE_PATH = "data/csv_files/combined_cleaned_embeddings.npy"
EMBEDDINGS_TSV_FILE_PATH = "data/csv_files/combined_cleaned_embeddings.tsv"


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
            input_file_path = os.path.join(PRIMARY_DIR_PATH, file.filename)

            with open(input_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            if file_ext.lower() == ".csv":
                chunks = process_txt(input_file_path)
                output_file_primary_csv_path = os.path.join(PRIMARY_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                output_file_cleaned_csv_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_primary_csv_path)
                processed_files.append(file.filename)
                process_data_embedded(output_file_primary_csv_path, output_file_cleaned_csv_path)

            elif file_ext.lower() == ".txt":
                chunks = process_txt(input_file_path)
                output_file_primary_csv_path = os.path.join(PRIMARY_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                output_file_cleaned_csv_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_primary_csv_path)
                processed_files.append(file.filename)
                process_data_embedded(output_file_primary_csv_path, output_file_cleaned_csv_path)

            elif file_ext.lower() == ".pdf":
                chunks = process_pdf(input_file_path)
                output_file_primary_csv_path = os.path.join(PRIMARY_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                output_file_cleaned_csv_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_primary_csv_path)
                processed_files.append(file.filename)
                process_data_embedded(output_file_primary_csv_path, output_file_cleaned_csv_path)

            elif file_ext.lower() == ".docx":
                chunks = process_docx(input_file_path)
                output_file_primary_csv_path = os.path.join(PRIMARY_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                output_file_cleaned_csv_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_primary_csv_path)
                processed_files.append(file.filename)
                process_data_embedded(output_file_primary_csv_path, output_file_cleaned_csv_path)

            elif file_ext.lower() == ".json":
                chunks = process_json(input_file_path)
                output_file_primary_csv_path = os.path.join(PRIMARY_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                output_file_cleaned_csv_path = os.path.join(CLEANED_CSV_DIR_PATH, f"{file_name}_chunks.csv")
                save_chunks_to_csv(chunks, output_file_primary_csv_path)
                processed_files.append(file.filename)
                process_data_embedded(output_file_primary_csv_path, output_file_cleaned_csv_path)

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        # Об'єднання неочищених файлів
        unprocessed_files = [f for f in os.listdir(PRIMARY_CSV_DIR_PATH) if f.endswith(".csv")]
        unprocessed_dfs = [pd.read_csv(os.path.join(PRIMARY_CSV_DIR_PATH, uf)) for uf in unprocessed_files]

        if unprocessed_dfs:
            unprocessed_combined_df = pd.concat(unprocessed_dfs, ignore_index=True)
            unprocessed_combined_df = ensure_unique_ids(unprocessed_combined_df, "chunk_id")
            unprocessed_combined_df.to_csv(PRIMARY_COMBINED_FILE_CSV_PATH, index=False)

        # Об'єднання очищених файлів
        cleaned_files = [f for f in os.listdir(CLEANED_CSV_DIR_PATH) if f.endswith(".csv")]
        dfs = [pd.read_csv(os.path.join(CLEANED_CSV_DIR_PATH, cf)) for cf in cleaned_files]

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = ensure_unique_ids(combined_df, "chunk_id")
            combined_df.to_csv(COMBINED_FILE_CSV_PATH, index=False)

            preprocess_and_generate_embeddings(COMBINED_FILE_CSV_PATH, EMBEDDINGS_FILE_PATH, EMBEDDINGS_TSV_FILE_PATH)
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
        print("Checking required files...")
        if not os.path.exists(BM25PUS_INDEX_FILE_PATH):
            raise HTTPException(status_code=404, detail="BM25 index file not found.")
        print("BM25 index file exists.")
        if not os.path.exists(TFIDF_INDEX_FILE_PATH):
            raise HTTPException(status_code=404, detail="TF-IDF index file not found.")
        print("TF-IDF index file exists.")
        if not os.path.exists(EMBEDDINGS_FILE_PATH):
            raise HTTPException(status_code=404, detail="Embeddings file not found.")
        print("Embeddings file exists.")
        if not os.path.exists(COMBINED_FILE_CSV_PATH):
            raise HTTPException(status_code=404, detail="Combined CSV file not found.")
        print("Combined CSV file exists.")

        print("Loading BM25 index...")
        bm25 = load_bm25_index(BM25PUS_INDEX_FILE_PATH)
        print("BM25 index loaded.")
        if not bm25 or not hasattr(bm25, "get_scores"):
            raise HTTPException(status_code=500, detail="BM25 index is invalid or corrupted.")
        print("BM25 index is valid.")

        print("Loading TF-IDF index...")
        vectorizer, tfidf_matrix = load_tfidf_index(TFIDF_INDEX_FILE_PATH)
        print("TF-IDF index loaded.")

        print("Loading embeddings...")
        document_embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)
        print("Embeddings loaded.")
        if document_embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings are not available.")
        print("Embeddings are valid.")

        print("Loading combined CSV data...")
        df = pd.read_csv(COMBINED_FILE_CSV_PATH)
        print("Combined CSV loaded.")
        if df.empty or "content" not in df.columns:
            raise HTTPException(status_code=500, detail="Combined file is missing or invalid.")
        print("Combined CSV file is valid.")

        print("Loading primary combined CSV data...")
        df_primary = pd.read_csv(PRIMARY_COMBINED_FILE_CSV_PATH)
        print("Primary combined CSV loaded.")
        if df_primary.empty or "content" not in df_primary.columns:
            raise HTTPException(status_code=500, detail="Primary combined file is missing or invalid.")
        print("Primary combined CSV file is valid.")

        print("Preprocessing query...")
        cleaned_query = preprocess_text_embedded(query.query)
        print(f"Cleaned query: {cleaned_query}")
        tokenized_query = cleaned_query.split()
        print(f"Tokenized query: {tokenized_query}")

        if not tokenized_query:
            raise HTTPException(status_code=400, detail="Query is empty or invalid after preprocessing.")

        print("Calculating BM25 scores...")
        bm25_scores = bm25.get_scores(tokenized_query)
        print(f"BM25 scores: {bm25_scores}")

        print("Calculating TF-IDF scores...")
        tfidf_scores = get_tfidf_scores(query.query, vectorizer, tfidf_matrix)
        print(f"TF-IDF scores: {tfidf_scores}")

        print("Generating query embedding...")
        query_embedding = model.encode([query.query], show_progress_bar=False)[0]
        print(f"Query embedding: {query_embedding}")

        print("Calculating similarity scores...")
        similarity_scores = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        print(f"Similarity scores: {similarity_scores}")

        print("Combining scores with weighted voting...")
        combined_scores = weighted_voting(bm25_scores, tfidf_scores, similarity_scores, alpha=0.4, beta=0.3, gamma=0.3)
        print(f"Combined scores: {combined_scores}")

        print("Selecting top documents...")
        top_k = 5
        top_k_indices = combined_scores.argsort()[-top_k:][::-1]
        print(f"Top indices: {top_k_indices}")

        ranked_documents = [
            {
                "id": int(df_primary.iloc[idx]["chunk_id"]),
                "score": float(combined_scores[idx]),
                "content": df_primary.iloc[idx]["content"]
            }
            for idx in top_k_indices
        ]
        print(f"Ranked documents: {ranked_documents}")

        print("Generating response...")
        context = ' '.join(doc['content'] for doc in ranked_documents)
        print(f"Context for response: {context}")
        assistant_response = generate_response(question=query.query, context=context)
        print(f"Assistant response: {assistant_response}")

        print("Returning results...")
        return {
            "query": query.query,
            "ranked_documents": ranked_documents,
            "assistant_response": assistant_response
        }

    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: {fnfe}")
        raise HTTPException(status_code=404, detail="Required file not found.")
    except Exception as e:
        print(f"Exception: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")

class PredictionRequest(BaseModel):
    date: str = Field(..., description="Date in format DD.MM.YYYY")
    event: str = Field(..., description="Event type: None, Holiday, Daily event, Promotions")
    model: str = Field(..., description="Model name: XGBoost, CatBoost, LightGBM, LinearRegression, DecisionTree")

    @field_validator("date")
    def validate_date_format(cls, value):
        try:
            datetime.strptime(value, "%d.%m.%Y")
            return value
        except ValueError:
            raise ValueError("Invalid date format. Use DD.MM.YYYY")

@app.post("/predict")
async def run_prediction(request: PredictionRequest):
    """
    Ендпоінт для запуску прогнозування на основі обраних параметрів.
    Параметри:
      - date: дата у форматі DD.MM.YYYY
      - event: None, Holiday, Daily event, Promotions
      - model: XGBoost, CatBoost, LightGBM, LinearRegression, DecisionTree
    """
    # Маппінг event у числові значення
    event_mapping = {
        "None": 0,
        "Holiday": 1,
        "Daily event": 2,
        "Promotion": 3
    }

    model_name_mapping = {
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "linearregression": "LinearRegression",
        "decisiontree": "DecisionTree"
    }

    # Перевірка коректності event
    if request.event not in event_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event}")

    # Перевірка коректності моделі
    model_name = request.model.lower()
    if model_name not in model_name_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {request.model}")

    # Конвертуємо дату у формат YYYY-MM-DD
    parsed_date = datetime.strptime(request.date, "%d.%m.%Y").strftime("%Y-%m-%d")

    model_class_name = model_name_mapping[model_name]
    module_path = f"edamonia_backend.logic.train.prediction.{model_class_name}"

    try:
        # Генерація тестових даних на основі event
        dataset_path = os.path.abspath("data/datasets")
        test_data = generate_10_data(parsed_date, event_mapping[request.event])
        test_data.to_csv(f"{dataset_path}/10_rows.csv", index=False)

        # Імпортуємо файл predict.py динамічно
        module = importlib.import_module(module_path)

        results = module.train(event_mapping[request.event], dataset_path)
        return {
            "message": f"Prediction successfully executed using {results['model_name']}",
            "date": parsed_date,
            "event": request.event,
            "parameters": results.get("parameters"),  # None для LinearRegression
            "cv_metrics": results.get("cv_metrics"),  # None для LinearRegression
            "test_metrics": results.get("test_metrics"),
        }

    except ModuleNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prediction module '{model_class_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.get("/download-result-table")
async def download_table(model: str):
    """
    Ендпоінт для завантаження таблиці результатів прогнозування.
    Параметри:
      - model: Назва моделі (CatBoost, XGBoost, DecisionTree, LightGBM, LinearRegression)
    """
    model_mapping = {
        "catboost": "CatBoost_predict.csv",
        "xgboost": "XGBoost_predict.csv",
        "decisiontree": "DecisionTree_predict.csv",
        "lightgbm": "LightGBM_predict.csv",
        "linearregression": "LinearRegression_predict.csv"
    }

    # Перевірка коректності моделі
    model_name = model.lower()
    if model_name not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model}")

    # Формуємо шлях до файлу
    file_path = os.path.join("edamonia_backend", "logic", "train", "prediction_results", model_mapping[model_name])

    # Перевірка наявності файлу
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Повертаємо файл
    return FileResponse(path=file_path, filename=model_mapping[model_name], media_type="application/csv")


@app.get("/download-test-results")
async def download_table(model: str):
    """
    Ендпоінт для завантаження таблиці результатів прогнозування.
    Параметри:
      - model: Назва моделі (CatBoost, XGBoost, DecisionTree, LightGBM, LinearRegression)
    """
    model_mapping = {
        "catboost": "CatBoost_test_predictions.csv",
        "xgboost": "XGBoost_test_predictions.csv",
        "decisiontree": "DecisionTree_test_predictions.csv",
        "lightgbm": "LightGBM_test_predictions.csv",
        "linearregression": "LinearRegression_test_predictions.csv"
    }

    # Перевірка коректності моделі
    model_name = model.lower()
    if model_name not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model}")

    # Формуємо шлях до файлу
    file_path = os.path.join("edamonia_backend", "logic", "train", "prediction_results", model_mapping[model_name])

    # Перевірка наявності файлу
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Повертаємо файл
    return FileResponse(path=file_path, filename=model_mapping[model_name], media_type="application/csv")


@app.delete("/delete-file/", summary="Delete a dataset and update indices")
async def delete_file(file_name: str = Query(..., description="Name of the file to delete")):
    """
    Видаляє файл із папок primary_csv і cleaned_csv, а також оновлює комбінований файл,
    BM25, TF-IDF індекси та ембеддінги.
    """
    try:
        # Перевіряємо наявність файлу в папці primary_csv
        primary_file_path = os.path.join(PRIMARY_CSV_DIR_PATH, file_name)
        if not os.path.exists(primary_file_path):
            raise HTTPException(status_code=404, detail=f"File not found in primary_csv: {file_name}")

        # Видаляємо файл із primary_csv
        os.remove(primary_file_path)

        # Перевіряємо наявність відповідного очищеного файлу в cleaned_csv
        cleaned_file_name = f"{os.path.splitext(file_name)[0]}_cleaned.csv"
        cleaned_file_path = os.path.join(CLEANED_CSV_DIR_PATH, cleaned_file_name)
        if os.path.exists(cleaned_file_path):
            os.remove(cleaned_file_path)

        # Оновлюємо комбінований файл
        cleaned_files = [f for f in os.listdir(CLEANED_CSV_DIR_PATH) if f.endswith(".csv")]
        dfs = [pd.read_csv(os.path.join(CLEANED_CSV_DIR_PATH, cf)) for cf in cleaned_files]

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = ensure_unique_ids(combined_df, "news_id")
            combined_df.to_csv(COMBINED_FILE_CSV_PATH, index=False)

            # Генеруємо нові ембеддінги
            preprocess_and_generate_embeddings(COMBINED_FILE_CSV_PATH, EMBEDDINGS_FILE_PATH)

            # Оновлюємо BM25 і TF-IDF індекси
            reindex_bm25(COMBINED_FILE_CSV_PATH, BM25PUS_INDEX_FILE_PATH)
            reindex_tfidf(COMBINED_FILE_CSV_PATH, TFIDF_INDEX_FILE_PATH)

            return {"message": f"File {file_name} deleted successfully and indices updated."}
        else:
            # Якщо немає очищених файлів, видаляємо комбінований файл, індекси та ембеддінги
            if os.path.exists(COMBINED_FILE_CSV_PATH):
                os.remove(COMBINED_FILE_CSV_PATH)
            if os.path.exists(BM25PUS_INDEX_FILE_PATH):
                os.remove(BM25PUS_INDEX_FILE_PATH)
            if os.path.exists(TFIDF_INDEX_FILE_PATH):
                os.remove(TFIDF_INDEX_FILE_PATH)
            if os.path.exists(EMBEDDINGS_FILE_PATH):
                os.remove(EMBEDDINGS_FILE_PATH)

            return {"message": f"File {file_name} deleted successfully. No more data to process."}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the file: {str(e)}")


@app.get("/datasets/", summary="Get all available datasets", response_description="List of available datasets")
async def get_datasets():
    """
    Отримує список всіх доступних датасетів у папці PRIMARY_CSV_DIR_PATH.
    """
    try:
        # Отримання списку файлів у папці
        datasets = [f for f in os.listdir(PRIMARY_CSV_DIR_PATH) if os.path.isfile(os.path.join(PRIMARY_CSV_DIR_PATH, f))]

        if not datasets:
            return {"message": "No datasets found."}

        return {"datasets": datasets}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving datasets: {str(e)}")
