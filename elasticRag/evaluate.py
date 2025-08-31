import json
import requests
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score


model = SentenceTransformer('all-mpnet-base-v2')

# URL of your Flask app
url = "http://localhost:8080/answer"  

def calculate_cosine_similarity(answer1, answer2):
    embeddings = model.encode([answer1, answer2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def evaluate_model(test_file, output_file):
    total_queries = 0
    correct_answers = 0
    total_bleu = 0.0
    total_wer = 0.0
    total_cosine = 0.0
    total_bert_precision = 0.0
    total_bert_recall = 0.0
    total_bert_f1 = 0.0
    metrics = []

    with open(test_file, 'r') as f:
        i = 0
        for line in f:
            i += 1
            if i % 10 == 0: print(f"Done evaluating on {i} samples")
            total_queries += 1
            data = json.loads(line)
            query = data['Question']
            expected_answer = data['Answer']

            
            response = requests.post(url, json={"query": query})
            if response.status_code == 200:
                response_data = response.json()
                model_answer = response_data.get('response')

                
                bleu_score = sentence_bleu([expected_answer.split()], model_answer.split())
                total_bleu += bleu_score


                word_error_rate = wer(expected_answer, model_answer)
                total_wer += word_error_rate

                cosine_sim = calculate_cosine_similarity(expected_answer, model_answer)
                total_cosine += cosine_sim

                P, R, F1 = bert_score([model_answer], [expected_answer], lang='en')
                bert_precision = P.mean().item()  
                bert_recall = R.mean().item()      
                bert_f1 = F1.mean().item()         

                total_bert_precision += bert_precision
                total_bert_recall += bert_recall
                total_bert_f1 += bert_f1

                
                metrics.append({
                    "query": query,
                    "expected": expected_answer,
                    "model": model_answer,
                    "BLEU Score": bleu_score,
                    "Word Error Rate": word_error_rate,
                    "Cosine Similarity": cosine_sim,
                    "BERT Precision": bert_precision,
                    "BERT Recall": bert_recall,
                    "BERT F1 Score": bert_f1,
                })

                

            else:
                print(f"Error: {response.status_code} for query: {query}")

    
    avg_bleu = total_bleu / total_queries if total_queries > 0 else 0
    avg_wer = total_wer / total_queries if total_queries > 0 else 0
    avg_cosine = total_cosine / total_queries if total_queries > 0 else 0
    avg_bert_precision = total_bert_precision / total_queries if total_queries > 0 else 0
    avg_bert_recall = total_bert_recall / total_queries if total_queries > 0 else 0
    avg_bert_f1 = total_bert_f1 / total_queries if total_queries > 0 else 0


    with open(output_file, 'w') as out_f:
        for metric in metrics:
            out_f.write(f"Query: {metric['query']}\n")
            out_f.write(f"Expected: {metric['expected']}\n")
            out_f.write(f"Model: {metric['model']}\n")
            out_f.write(f"BLEU Score: {metric['BLEU Score']:.4f}\n")
            out_f.write(f"Word Error Rate: {metric['Word Error Rate']:.4f}\n")
            out_f.write(f"Cosine Similarity: {metric['Cosine Similarity']:.4f}\n")
            out_f.write(f"BERT Precision: {metric['BERT Precision']:.4f}\n")
            out_f.write(f"BERT Recall: {metric['BERT Recall']:.4f}\n")
            out_f.write(f"BERT F1 Score: {metric['BERT F1 Score']:.4f}\n")
            out_f.write("\n")

    # Summary
    with open(output_file, 'a') as out_f:
        out_f.write(f"Total Queries: {total_queries}\n")
        out_f.write(f"Average BLEU Score: {avg_bleu:.4f}\n")
        out_f.write(f"Average Word Error Rate: {avg_wer:.4f}\n")
        out_f.write(f"Average Cosine Similarity: {avg_cosine:.4f}\n")
        out_f.write(f"Average BERT Precision: {avg_bert_precision:.4f}\n")
        out_f.write(f"Average BERT Recall: {avg_bert_recall:.4f}\n")
        out_f.write(f"Average BERT F1 Score: {avg_bert_f1:.4f}\n")

if __name__ == "__main__":
    evaluate_model("syntheticTest.jsonl", "syntheticTestResults.txt")  # Paths to your test file and output file
    evaluate_model("syntheticTestComplex.jsonl", "syntheticTestComplexResults.txt")  # Paths to your test file and output file
    evaluate_model("maskedTest.jsonl", "maskedTestResults.txt")  # Paths to your test file and output file
    