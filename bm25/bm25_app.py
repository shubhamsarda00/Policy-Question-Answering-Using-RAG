from flask import Flask, request, jsonify, render_template

from transformers import pipeline
import os
import pdfplumber
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import nltk
app = Flask(__name__)

nltk.download('punkt')
nltk.download('punkt_tab')
def getPdfs(path):
    files = os.listdir(path)
    return [os.path.join(path,file) for file in files if ".pdf" in file]

def extractTextFromPdf(pdfPath):
    with pdfplumber.open(pdfPath) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

pdfPath = 'DS Assignment/pdfs/'


pdfFiles = getPdfs(pdfPath)

documents = [extractTextFromPdf(pdf) for pdf in pdfFiles]

def chunkDocument(document, max_chunk_size=1000):
    sentences = sent_tokenize(document)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chunk_size:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

documentChunks = [chunkDocument(doc) for doc in documents]

tokenizedDocuments = [word_tokenize(document) for document in documents]

bm25Docs = BM25Okapi(tokenizedDocuments)

qa_model = pipeline("question-answering", model="deepset/roberta-large-squad2")

@app.route('/health', methods=['GET'])
def health_check():
   
    if not documents:  # Check if documents are loaded
        return jsonify({"status": "unhealthy", "reason": "No documents loaded"}), 503

    return jsonify({"status": "healthy", "ready": True}), 200


@app.route('/')
def form():
    return render_template('ui.html')

@app.route('/retrieve', methods=['POST'])
def retrieve():

    data = request.get_json()
    query = data.get('query')    
    if not query:
        return jsonify({"error": "Query is required"}), 400


    tokenizedQuery = word_tokenize(query)

    
    docScores = bm25Docs.get_scores(tokenizedQuery)

    
    topK = 3
    topKDocIndices = np.argsort(docScores)[::-1][:topK]
    topKDocuments = [documents[i] for i in topKDocIndices]
    sources = [pdfFiles[i] for i in topKDocIndices]
    return jsonify({
            "sources": sources
        })

@app.route('/answer', methods=['POST'])
def get_answer():
    
    if request.is_json:
        data = request.get_json()
        query = data.get('query')    
        if not query:
            return jsonify({"error": "Query is required"}), 400
    else:
        query = request.form.get('query')
    
    
    tokenizedQuery = word_tokenize(query)

#k1=1.5, b=0.75
    docScores = bm25Docs.get_scores(tokenizedQuery)

    
    topK = 3
    topKDocIndices = np.argsort(docScores)[::-1][:topK]
    topKDocuments = [documents[i] for i in topKDocIndices]
    sources = [pdfFiles[i] for i in topKDocIndices]

    
    allChunks = [chunk for docChunks in [documentChunks[i] for i in topKDocIndices] for chunk in docChunks]


    tokenizedChunks = [word_tokenize(chunk) for chunk in allChunks]


    bm25Chunks = BM25Okapi(tokenizedChunks)


    chunkScores = bm25Chunks.get_scores(tokenizedQuery)

    topN = 3
    topNChunkIndices = np.argsort(chunkScores)[::-1][:topN]
    topPassages = [allChunks[i] for i in topNChunkIndices]

   
    combinedPassage = " ".join(topPassages)


    response = qa_model(question=query, context=combinedPassage)

    # Return the answer and sources
    if request.is_json:
        return jsonify({
            "response": response['answer'],
            "sources": topPassages
        })
    else:
        return render_template('ui.html', query=query, answer=response['answer'], source=topPassages)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)