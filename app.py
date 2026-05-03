from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from agent_utils import extract_text, create_vectorstore, summarize_document, generate_mcqs, answer_question

app = Flask(__name__)
app.secret_key = '***************

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf','docx','txt','png','jpg','jpeg','csv','xlsx','xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

summary_data = ''
mcq_data = ''
answer_data = ''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global summary_data, mcq_data
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Unsupported file type')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        text = extract_text(filepath)
        if not text.strip():
            flash('No readable text found')
            return redirect(url_for('index'))

        create_vectorstore(text)
        summary_data = summarize_document()
        mcq_data = ''

        return render_template('chat.html', summary=summary_data, mcqs=mcq_data, answer='')

    except Exception as e:
        flash(str(e))
        return redirect(url_for('index'))

@app.route('/mcqs', methods=['POST'])
def mcqs():
    global mcq_data, summary_data, answer_data
    mcq_data = generate_mcqs()
    return render_template('chat.html', summary=summary_data, mcqs=mcq_data, answer=answer_data)

@app.route('/ask', methods=['POST'])
def ask():
    global answer_data, summary_data, mcq_data
    question = request.form.get('question','').strip()
    if not question:
        return render_template('chat.html', summary=summary_data, mcqs=mcq_data, answer='Enter a question.')

    answer_data = answer_question(question)
    return render_template('chat.html', summary=summary_data, mcqs=mcq_data, answer=answer_data)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
