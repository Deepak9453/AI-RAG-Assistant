import os
from transformers import pipeline

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------- EMBEDDING MODEL ---------------- #

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# ---------------- LLM MODEL ---------------- #

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    temperature=0.2,
    do_sample=False
)


# ---------------- VECTOR STORE ---------------- #

def create_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    vectorstore = FAISS.from_texts(
        chunks,
        embedding
    )

    os.makedirs("vectorstore", exist_ok=True)

    vectorstore.save_local("vectorstore")

    return vectorstore


def load_vectorstore():

    return FAISS.load_local(
        "vectorstore",
        embedding,
        allow_dangerous_deserialization=True
    )


# ---------------- SUMMARY ---------------- #

def summarize_document():

    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search("main topic", k=5)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Summarize the following document clearly in 5-6 sentences.

Document:
{context}
"""

    result = generator(prompt, max_new_tokens=200)

    return result[0]["generated_text"]


# ---------------- MCQ GENERATOR ---------------- #

def generate_mcqs():

    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search("important concepts", k=5)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a teacher creating quiz questions.

Create EXACTLY 5 MCQs from the text.

Rules:
- Each question must have 4 options
- Only one correct answer
- Questions should test understanding

Format EXACTLY like this:

Q1: Question
A. Option
B. Option
C. Option
D. Option
Answer: A

Q2: Question
A. Option
B. Option
C. Option
D. Option
Answer: B

Text:
{context}
"""

    result = generator(prompt, max_new_tokens=400)

    return result[0]["generated_text"]


# ---------------- QUESTION ANSWERING ---------------- #

def answer_question(question):

    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer clearly:
"""

    result = generator(prompt, max_new_tokens=200)

    return result[0]["generated_text"]