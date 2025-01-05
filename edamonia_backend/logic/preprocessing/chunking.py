import json

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document


def process_csv(file_path):
    df = pd.read_csv(file_path)

    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV файл повинен мати хоча б дві колонки!")

    chunks = df.apply(
        lambda row: ", ".join(f"{col} = {row[col]}" for col in df.columns),
        axis=1
    ).tolist()

    return chunks


def chunk_text_with_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
        ],
    )
    return text_splitter.split_text(text)

def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return chunk_text_with_langchain(text)

def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages)
    return chunk_text_with_langchain(text)

def process_docx(file_path):
    doc = Document(file_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    return chunk_text_with_langchain(text)

def process_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        text = json.dumps(data, indent=2)
    return chunk_text_with_langchain(text)

def save_chunks_to_csv(chunks, output_path):
    df = pd.DataFrame({"chunk_id": range(1, len(chunks) + 1), "content": chunks})
    df.to_csv(output_path, index=False)