from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session
import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import sent_tokenize

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download NLTK sentence tokenizer
nltk.download('punkt')

# Load SentenceTransformer model
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Global variables
doc_chunks = []
doc_embeddings = None

# ------------------ Utilities ------------------

def extract_text(filepath):
    """Extract full text from PDF using PyMuPDF."""
    doc = fitz.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()


def smart_chunk_text(text, max_tokens=200):
    """Chunk text based on complete sentences (not raw word count)."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    total_tokens = 0

    for sentence in sentences:
        tokens = sentence.split()
        if total_tokens + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            total_tokens = 0
        current_chunk.append(sentence)
        total_tokens += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def embed_chunks(chunks):
    """Create embeddings for all chunks."""
    return model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)


def refine_question(question):
    """Light preprocessing of the user question."""
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
    return question.lower().capitalize()


def get_best_answer(question, k=5, threshold=0.5):
    """Return the most relevant chunk(s) based on cosine similarity."""
    global doc_chunks, doc_embeddings

    if doc_embeddings is None or len(doc_chunks) == 0:
        return "No document uploaded."

    question = refine_question(question)
    q_emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity
    scores = util.cos_sim(q_emb, doc_embeddings)[0]
    top_indices = np.argsort(scores.cpu().numpy())[::-1]

    answers = []
    for idx in top_indices[:k]:
        if scores[idx] >= threshold:
            answers.append((doc_chunks[idx], float(scores[idx])))

    if not answers:
        return "Sorry, I couldn't find a confident answer in the document."

    return answers[0][0]  # Return best matching chunk


# ------------------ Routes ------------------

@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    global doc_chunks, doc_embeddings

    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # Extract and process PDF content
    text = extract_text(path)
    doc_chunks = smart_chunk_text(text)
    doc_embeddings = embed_chunks(doc_chunks)
    session['pdf_name'] = filename

    return redirect(url_for('chat'))


@app.route('/chat')
def chat():
    pdf_name = session.get('pdf_name', 'No file uploaded')
    return render_template('ask.html', pdf_name=pdf_name)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    answer = get_best_answer(question)
    return Response(answer, content_type='text/plain')


# ------------------ Main ------------------

if __name__ == '__main__':
    app.run(debug=True)
