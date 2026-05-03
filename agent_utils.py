import os
import io
import shutil
import zipfile
import pandas as pd
import PyPDF2
import docx
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_PATH = 'vectorstore'

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)


def generate_text(prompt):
    response = ollama.chat(
        model='llama3',
        messages=[{'role':'user','content':prompt}]
    )
    return response['message']['content']


def extract_text(filepath):
    filename = filepath.lower()
    text = ''

    if filename.endswith('.pdf'):
        with open(filepath,'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        if not text.strip():
            images = convert_from_path(filepath)
            for img in images:
                text += pytesseract.image_to_string(img)

    elif filename.endswith('.docx'):
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + '\n'

    elif filename.endswith('.txt'):
        with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
            text = f.read()

    elif filename.endswith(('.png','.jpg','.jpeg')):
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)

    elif filename.endswith('.csv'):
        df = pd.read_csv(filepath)
        text = df.to_string(index=False)

    elif filename.endswith(('.xlsx','.xls')):
        sheets = pd.read_excel(filepath, sheet_name=None)
        for name, df in sheets.items():
            text += f'\nSheet: {name}\n'
            text += df.to_string(index=False)

    return text


def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(text)

    if os.path.exists(VECTOR_PATH):
        shutil.rmtree(VECTOR_PATH)

    db = FAISS.from_texts(chunks, embedding)
    db.save_local(VECTOR_PATH)


def load_vectorstore():
    return FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)


def summarize_document():
    db = load_vectorstore()
    docs = db.similarity_search('main summary', k=5)
    context = '\n\n'.join([d.page_content for d in docs])

    prompt = f'''Summarize this document in 6 clear bullet points.\n\n{context}'''
    return generate_text(prompt)


def generate_mcqs():
    db = load_vectorstore()
    docs = db.similarity_search('important concepts', k=5)
    context = '\n\n'.join([d.page_content for d in docs])

    prompt = f'''Create 5 MCQs from the text below. Each must have 4 options and 1 correct answer.\n\n{context}'''
    return generate_text(prompt)


def answer_question(question):
    db = load_vectorstore()
    docs = db.similarity_search(question, k=8)
    context = '\n\n'.join([d.page_content for d in docs])

    prompt = f'''Use only the context below to answer clearly. If not found, say information not available.\n\nContext:\n{context}\n\nQuestion: {question}'''
    return generate_text(prompt)
