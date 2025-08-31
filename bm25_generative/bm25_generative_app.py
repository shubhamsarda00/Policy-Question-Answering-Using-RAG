from flask import Flask, request, jsonify, render_template
import os
import pdfplumber
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import openai
from openai import OpenAI
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
    allChunks = [chunk for docChunks in [documentChunks[i] for i in topKDocIndices] for chunk in docChunks]


    tokenizedChunks = [word_tokenize(chunk) for chunk in allChunks]


    bm25Chunks = BM25Okapi(tokenizedChunks)


    chunkScores = bm25Chunks.get_scores(tokenizedQuery)

    topN = 5
    topNChunkIndices = np.argsort(chunkScores)[::-1][:topN]
    topPassages = [allChunks[i] for i in topNChunkIndices]

    return jsonify({
            "sources": topPassages
        })
      
os.environ['OPENAI_API_KEY'] = 'TBD'
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)
def generateResponse(query,context):

    prompt = f"Context: {context}\n\nBased on the context, generate an answer for the following question: {query}"
    # prompt = f"Given the following document excerpt:\n\n\"{document}\"\n\nPlease generate a relevant question and answer based on the text."
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
    "role": "user",
    "content": prompt
    }
    # {"role": "user", "content": f"Given the following document excerpt:\n\n\"{chunk}\"\n\nPlease generate 5 relevant question and answer pairs based on the text in the following format as a list of dictionaries: [\{'Question':'..','Answer':'..'\},\{'Question':'..','Answer':'..'\}...]"}
    ],
    max_tokens=192,
        temperature=0.7,
        top_p=1,
        n=1,
        stop=None
    )

    return completion

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

    docScores = bm25Docs.get_scores(tokenizedQuery)

    
    topK = 3
    topKDocIndices = np.argsort(docScores)[::-1][:topK]
    topKDocuments = [documents[i] for i in topKDocIndices]
    sources = [pdfFiles[i] for i in topKDocIndices]

    
    allChunks = [chunk for docChunks in [documentChunks[i] for i in topKDocIndices] for chunk in docChunks]


    tokenizedChunks = [word_tokenize(chunk) for chunk in allChunks]


    bm25Chunks = BM25Okapi(tokenizedChunks)


    chunkScores = bm25Chunks.get_scores(tokenizedQuery)

    topN = 5
    topNChunkIndices = np.argsort(chunkScores)[::-1][:topN]
    topPassages = [allChunks[i] for i in topNChunkIndices]

   
    combinedPassage = " ".join(topPassages)


    response = generateResponse(query, combinedPassage)

    # Return the answer and sources
    if request.is_json:
        return jsonify({
            "response": response.choices[0].message.content,
            "sources": topPassages
        })
    else:
        return render_template('ui.html', query=query, answer=response.choices[0].message.content, source=topPassages)

if __name__ == "__main__":

    app.run(host='0.0.0.0', port=8000, debug=True)
