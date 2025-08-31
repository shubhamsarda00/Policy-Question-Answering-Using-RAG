
from flask import Flask, request, jsonify, render_template
import pdfplumber
import os
import numpy as np
from elasticsearch import Elasticsearch
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)


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
def chunkDocument(document, max_chunk_size=1000, overlap=2):
    sentences = sent_tokenize(document)
    chunks = []
    chunk = ""
    overlapContext = []
    for i,sentence in enumerate(sentences):
        if len(chunk) + len(sentence) < max_chunk_size:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            overlapContext = sentences[max(0,i-overlap):i]
            chunk = " ".join(overlapContext) + sentence
            # if len(chunks)==2:
            #     print(chunks[-1])
            #     print()
            #     print(overlapContext)
            #     print()
            #     print(chunk)
    if chunk:
        chunks.append(chunk.strip())
    return chunks

documentChunks = [chunkDocument(doc) for doc in documents]



ngrokUrl = "http://localhost:9200"
es = Elasticsearch(ngrokUrl)
indexName = "my_index"
es.indices.delete(index=indexName, ignore=[400, 404])
def createIndex(indexName):
    es.indices.create(
        index=indexName,
        body={
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384  
                    },
                    "metadata": {
                        "properties": {
                            "chunk_id": { "type": "text" },
                            "doc_id": { "type": "text" },
                            "text": { "type": "text" }
                        }
                    }
                }
            }
        }
    )


createIndex(indexName)



# openAiEmbedding = OpenAIEmbeddings()
model = SentenceTransformer('all-MiniLM-L6-v2')

def generateEmbedding(text):
    return model.encode(text)

def indexDocumentChunks(indexName, documentChunks):
    for docId,chunks in enumerate(documentChunks):
        for chunkId, chunk in enumerate(chunks):
            vector = generateEmbedding(chunk)
            body = {
                'vector': vector.tolist(),  
                'metadata': {
                    'chunk_id': f'{chunkId}',
                    'doc_id': f"{docId}",
                    'text': chunk
                }
            }
            es.index(index=indexName, id=f'{docId}_{chunkId}', body=body)



indexDocumentChunks('my_index', documentChunks)


def searchSimilarChunks(indexName, queryText, size=5):
    queryVector = generateEmbedding(queryText)
    # body = {
    #     "query": {
    #         "knn": {
    #             "vector": {
    #                 "vector": queryVector.tolist(),
    #                 "k": size
    #             }
    #         }
    #     }
    # }
    body = {
    "size": size,  
    "_source": True,  
    "query": {
        "script_score": {
            "query": {
                "match_all": {}  
            },
            "script": {
                "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                "params": {
                    "queryVector": queryVector.tolist()
                }
            }
        }
    }
}
    response = es.search(index=indexName, body=body)
    return response['hits']['hits']


@app.route('/')
def form():
    return render_template('ui.html')

@app.route('/retrieve', methods=['POST'])
def retrieve():

    data = request.get_json()
    query = data.get('query')    
    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = searchSimilarChunks('my_index', query)
    
    return jsonify({
            "sources": results
        })

os.environ['OPENAI_API_KEY'] = 'sk-xpZI1AehtXlMCE7G1ZzhIfHPTQ6F4NXQrpAwkj5zKaT3BlbkFJaHrJInF4uLx_8UMrEQpDjRNuSBdUH8MqSymSByhT8A'
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)



print("INIT_DONE")

def generateResponse(query,results):

    context = ' '.join([result["_source"]["metadata"]["text"] for result in results[:]])
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
        #print(query)
        if not query:
            return jsonify({"error": "Query is required"}), 400
    else:
        query = request.form.get('query')
    #print(query)
    results = searchSimilarChunks('my_index', query)
    response = generateResponse(query, results)

    sources = [x["_source"]["metadata"]["text"] for x in results]
    if request.is_json:
        return jsonify({
            "response": response.choices[0].message.content,
            "sources": sources})
    else:
        return render_template('ui.html', query=query, answer=response.choices[0].message.content, source=sources)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)