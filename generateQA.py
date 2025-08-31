import os
import pdfplumber
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import openai
from openai import OpenAI
import json
import nltk

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

pdfPath = '/content/drive/MyDrive/Simpplr Assignment/DS Assignment/pdfs/'



pdfFiles = getPdfs(pdfPath)

documents = [extractTextFromPdf(pdf) for pdf in pdfFiles]

def chunkDocument(document, max_chunk_size=500):
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

os.environ['OPENAI_API_KEY'] = 'TBD'
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

def generate_qa_gpt(document):
    prompt = f"Given the following document excerpt:\n\n\"{document}\"\n\nPlease generate a relevant question and answer based on the text."
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
    "role": "user",
    "content": (
        f"Given the following document excerpt:\n\n\"{document}\"\n\n"
        "Please generate 10 relevant question and answer pairs based on the text. Adhere to the following output format as a list of dictionaries : "
        "[{'Question':'..','Answer':'..'},{'Question':'..','Answer':'..'},...]"
    )
    }
    # {"role": "user", "content": f"Given the following document excerpt:\n\n\"{chunk}\"\n\nPlease generate 5 relevant question and answer pairs based on the text in the following format as a list of dictionaries: [\{'Question':'..','Answer':'..'\},\{'Question':'..','Answer':'..'\}...]"}
    ],
    max_tokens=1536,
        temperature=0.7,
        top_p=1,
        n=1,
        stop=None
    )


    return completion
    
qaPairs = []
responses = []
for i in range(len(documents)):
    response = generate_qa_gpt(documents[i])
    text=response.choices[0].message.content
    start_index = text.find('[')
    end_index = text.rfind(']') + 1  
    qa_pairs_text=""
    
    if start_index != -1 and end_index != -1:
        qa_pairs_text = text[start_index:end_index]
    try:
        currPairs = eval(qa_pairs_text)  # Use json.loads for safety
        qaPairs.append(currPairs)
    except:
        print(f"Error processing chunk {i}")
        responses.append(text)
    
with open('syntheticTest.jsonl', 'w',encoding= 'utf-8') as jsonl_file:
    for item in flattenedQaPairs:
        jsonl_file.write(json.dumps(item) + '\n')
       
       
import random
used= set()

def generateNewQaPair(qaPairs):
    while True:
        indices = random.sample(range(len(qaPairs)), 2)
        indexPair = tuple(sorted(indices))  # Sort to avoid duplicates
        
        if indexPair not in used:
            used.add(indexPair)
            pair1, pair2 = qaPairs[indices[0]], qaPairs[indices[1]]
            newQuestion = pair1['Question'] + " " + pair2['Question']
            newAnswer = pair1['Answer'] + " " + pair2['Answer']
            return {"Question": newQuestion, "Answer": newAnswer}

newQaPairs = []
for _ in range(100):  
    newQaPair = generateNewQaPair(flattenedQaPairs)
    newQaPairs.append(newQaPair)

with open('syntheticTestComplex.jsonl', 'w',encoding= 'utf-8') as jsonl_file:
    for item in newQaPairs:
        jsonl_file.write(json.dumps(item) + '\n')
        
        
import random
from nltk.tokenize import word_tokenize  # Make sure to include this if you haven't

def createMaskedQaPairs(text):
    words = word_tokenize(text)
    indicesToMask = random.sample(range(len(words)), len(words)//2)

    maskedWords = words.copy()
    for index in indicesToMask:
        maskedWords[index] = "----"

    maskedQuestion = ' '.join(maskedWords)
    instruction = ' In the given above passage, some words have been masked with "----". Fill in the masked/missing words based on the initial context.'
    maskedQuestion += instruction

    originalAnswer = text
    qaPair = {
        "Question": maskedQuestion,
        "Answer": originalAnswer
    }
    return qaPair



maskedQaPairs = []

documentChunksFlat = [item for sublist in documentChunks for item in sublist]
for chunk in documentChunksFlat:
    maskedQaPair = createMaskedQaPairs(chunk)
    maskedQaPairs.append(maskedQaPair)



with open('maskedTest.jsonl', 'w',encoding= 'utf-8') as jsonl_file:
    for item in maskedQaPairs:

        jsonl_file.write(json.dumps(item) + '\n')
