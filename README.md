# Policy Q/A Using RAG

## Task Overview
The objective is to build a Question Answering (QA) system that can accurately answer policy-related queries (e.g., "How many personal leaves do I have?") by extracting answers strictly from provided policy documents (PDFs). 

**Objectives:**

- Develop 2–3 different solution approaches: Justify each with pros, cons, and reasoning.
- Define evaluation criteria: Compare solutions based on accuracy, latency, relevance, etc.
- Build a REST API
  - Accepts user_query as input.
  - Returns:
    - response: the generated answer.
    - sources: relevant document chunks used to form the answer.
- Containerize the service using Docker.
- Write production-quality code with proper structure and unit tests.

## Approach

Details can be found in **AssigmentDetails.docx**

### 1. BM25 + Extractive Q&A

We first read all the pdfs page wise and extract the text from the same. Now we create chunks of size max size 1000 characters for each pdf since it way more efficient to retrieve relevant chunks rather than the complete pdfs later down the pipeline for question answer. Chunking increases performance as well since it allows us to focus only on the relevant sections and avoids confusing the Q&A model with garbage context. For chunking, we first tokenize the document text into sentences and then iteratively create chunks from these sentences maintaining the constraint of 1000 characters. We then feed the word tokenized documents into the bm25Okapi. It does the required preprocessing (inverted index computation) of the source documents for bm25 scoring/retrieval. All this is done for the preprocessing part.
In order to answer a new query, we tokenize the query and find the bm25 score wrt each document and sort them in descending order. We generate a list of chunks for the top3 documents and again use bm25 retrieval to fetch the most relevant chunks. Top 3 chunks are retained for usage as context to answer the query. We use a pre-trained transformer model (roberta-large-squad2) to answer the query by analyzing and extracting answer from the combined relevant passages/chunks from the documents.
We’ve defined the main flask endpoint as **“/answer”**. It takes the input as jsonified “query” and uses the strategy discussed above to return the “response” and the “source” chunks retrieved. It can also handle and process inputs from the web url api. The flask app is made accessible on the port 5000 of the localhost address. 
All relevant code lies in **bm25_app.py**

---

### 2. ElasticSearch RAG

Similar to previous approach, we first read all the pdfs page wise and extract the text from the same. Then we create chunks of size max size 1000 characters from each document. For chunking, we first tokenize the document text into sentences and then iteratively create chunks from these sentences maintaining the constraint of 1000 characters. In order to maintain continuity, we also include an overlap of 2 sentences between successive chunks.
Then we initialize our elastic search index with parameters such embedding dimension, meta data for chunks such as chunk_id, doc_id etc. We then index all the chunks using the metadata and the corresponding sentence embedding. The embedding is generated using sentence-transformer library. We use a relatively smaller model “all-MiniLM-L6-v2” for quick embedding computations. All this is done as part of preprocessing. 
In order to answer a new query, we find the embedding of the query use elastic search to find top k nearest /most similar chunks from the vector database. We use cosine similarity for to compare 2 embeddings. Top 5 chunks are retained for usage as context to answer the query. We use a gpt4o-mini to answer the query. We augment the prompt with the relevant chunk data and instruct the model to answer the query based on the context provided.
We’ve defined the main flask endpoint as **“/answer”**. It takes the input as jsonified “query” and uses the strategy discussed above to return the “response” and the “source” chunks retrieved. It can also handle and process inputs from the web url api. The flask app is made accessible on the port 8080 of the localhost address. 
All relevant code lies in **elasticRag_app.py**

---

### 3. BM25 RAG (Hybrid)

This is a hybrid approach which employs BM25 retrieval for fetch the relevant chunks similar to approach 1. Then, we augment the prompt/query with relevant context and instruct a generative model (gpt40-mini) to answer the question based on the context provided.
All relevant code lies in **bm25_generative_app.py**

---

### Unit Testing

In order to test the validity of the Q&A model/code, we form basic unit tests during docker build process itself so that in case of failure, the process stops there itself. We’ve used pytest to configure the unit tests. We have 2 tests for both the Q&A models:

a)	Retrieval Test: Invokes the **'/retrieve'** flask endpoint to retrieve top “k” relevant document chunks based on a dummy query. We’ve added assert checks to ensure that retrieval call to the endpoint is successful. Further check the number of documents retrieved as the as the “k” set for the model.

b)	End to End Test: Invokes the **'/answer'** flask endpoint to retrieve the response for a dummy query. Here retrieval as well as Q&A happens hence being a more complete check of correctness. We’ve added assert checks to ensure that answer call to the endpoint is successful. Further check that the output is a json and has the “response” and “sources” as keys to check the correctness of the output.

Tests are defined in **test.py**


## Evaluation Strategy

### Test Set Generation
 In order to evaluate the model, we need to generate synthetic question answer pairs from the pdfs. We’ve used 2 strategies for the same:

- **Using Generative Model:** We pass the complete document text as context along with instructions to generate 10 question answer pairs from the excerpt. We do this for all documents so we can generate 100 QA pairs in total. Here we avoid chunking, because we want questions randomly sampled from different parts of the document and possibly having answers in different chunks as well. For generating QA pairs, we use GPT4-O, a superior model relative to the one we used for Q&A in RAG approach. This ensures that the test set quality is high. We instruct the model to generate QA pairs in json format for easy automated extraction. These question answer pairs are present in syntheticTest.jsonl 
Now using the QA pairs generated, we also create “complex” QA pairs by randomly selecting any 2 pairs from 100 and concatenating the question and answer parts respectively to generate the new QA pair. This helps us create complex unrelated questions from possibly different documents and hence tests the retrieval as well as Q&A efficiency of the model. These question answer pairs are present in syntheticTestComplex.jsonl
- **Masking Document Chunks:** A cheaper way to create a test without using generative models is to create chunks of document and randomly mask 50% of the words. The masked chunk + an instruction to fill in the missing words forms the query. The original chunk forms the ground truth answer. We’ve used chunks of max size 500 characters; hence total 107 QA pairs were created for this scenario. These question answer pairs are present in maskedTest.jsonl
  
All relevant code lies in **generateQA.py**

### Metrics Used
- **BLEU Score**  
- **Word Error Rate (WER)**  
- **Cosine Similarity** (vs superior model embeddings)  
- **BERTScore** (Precision, Recall, F1)  

---

### Results

| **Metric**         | **BM25 + Extr. Q&A** | **ElasticSearch RAG** | **BM25 RAG** |
|---------------------|-----------------------|------------------------|--------------|
| BLEU (Synthetic)   | 0.065                | 0.30                  | 0.31         |
| BLEU (Complex)     | 0.021                | 0.231                 | 0.307        |
| BLEU (Masked)      | 0.0057               | 0.382                 | 0.389        |
| Cosine Sim. (Synth)| 0.47                 | 0.836                 | 0.838        |
| Cosine Sim. (Masked)| 0.24                | 0.841                 | 0.849        |
| BERTScore F1 (Synth)| 0.85                | 0.924                 | 0.925        |
| BERTScore F1 (Complex)| 0.83              | 0.90                  | 0.917        |

**Observations**:  
- **BM25 + Extractive**: Too short answers → low BLEU/WER despite precision.  
- **ElasticSearch RAG**: Stronger overall, but WER inflated due to long outputs.  
- **BM25 RAG**: Best on **complex queries**, slight edge over ElasticSearch RAG.  

---


## Instructions for running the code

### BM25:

1) Create Docker Image: 
- `cd bm25/`
- `docker build -t bm25-api .`: This will install the dependencies, start up the application, perform the unit tests and in case success, perform the evaluations as well. Post successful running of this, a docker image will be created 

2) Extract evaluation text files:
- `docker run -it bm25-api /bin/bash`: This will create and start a new container from a Docker image. It will initiate an interactive pseudo terminal through which you explore files in the image.
- `docker ps`: This will give you a list of containers, where you can find the container id for your image
- `docker cp <container_id>:/app/evaluation_results.txt ./evaluation_results.txt`: Use this to copy required files from app directory of the image to current directory of the terminal
- `exit`: This will exit the terminal
3) Start the container:
- `docker run -p 5000:5000 bm25-api`: This will create and start a container with the specified docker image. The first port is the local port and second is the port inside container where application is running

For other strategies, the steps are the same but the image name, main directory and port change.

### Elastic RAG:

1) `cd elasticRag/`
    `docker build -t elastic_rag-api .`
2) `docker run -it elastic_rag-api /bin/bash`
3) `docker ps`
4) `docker cp <container_id>:/app/evaluation_results.txt ./evaluation_results.txt`
5) `docker run -p 8080:8080 elastic_rag-api`


### BM25 RAG:
1) `cd bm25_generative/`
    `docker build -t bm25_generative-api .`
2) `docker run -it bm25_generative-api /bin/bash`
3) `docker ps`
4) `docker cp <container_id>:/app/evaluation_results.txt ./evaluation_results.txt`
5) `docker run -p 8000:8000 bm25_generative-api`

### Calling the Application
Once the container is up and running, you can hit the api in two ways:
1) Web Url Route:
Go to `/` endpoint i.e. `http://localhost:XXXX/` in your browser and type your query in the query box and click submit.
[![Web Url Call](https://i.postimg.cc/52BGmgPx/weburl-Call.png)](https://postimg.cc/MnGd6724)
2) Post Call:
Make post call to the `/answer` endpoint and provide the jsonified query as input. You can change the port `(XXXX)`in the command depending on the container you're running.
`curl -X POST http://localhost:XXXX/answer -H "Content-Type: application/json" -d "{\"query\": \"How many personal leaves do I have?\"}"`
[![Post Call Output](https://i.postimg.cc/sxshdVYQ/postCall.png)](https://postimg.cc/8JX5LgnN)

For both the calls, you get the response and source chunks as the output.


