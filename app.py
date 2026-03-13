from flask import Flask, render_template, request
import PyPDF2
import docx

from agent_utils import (
    create_vectorstore,
    summarize_document,
    generate_mcqs,
    answer_question
)

app = Flask(__name__)

summary_data = ""
mcq_data = ""
answer_data = ""


# ---------------- FILE TEXT EXTRACTION ---------------- #

def extract_text(file, filename):

    text = ""

    if filename.endswith(".pdf"):

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    elif filename.endswith(".docx"):

        document = docx.Document(file)

        for para in document.paragraphs:
            text += para.text + "\n"

    return text


# ---------------- HOME PAGE ---------------- #

@app.route("/")
def index():

    return render_template("index.html")


# ---------------- FILE UPLOAD ---------------- #

@app.route("/upload", methods=["POST"])
def upload():

    global summary_data
    global mcq_data

    file = request.files["file"]

    text = extract_text(file, file.filename)

    create_vectorstore(text)

    summary_data = summarize_document()

    mcq_data = generate_mcqs()

    return render_template(
        "chat.html",
        summary=summary_data,
        mcqs=mcq_data,
        answer=""
    )


# ---------------- ASK QUESTION ---------------- #

@app.route("/ask", methods=["POST"])
def ask():

    global answer_data

    question = request.form["question"]

    answer_data = answer_question(question)

    return render_template(
        "chat.html",
        summary=summary_data,
        mcqs=mcq_data,
        answer=answer_data
    )


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000,
        debug=True
    )