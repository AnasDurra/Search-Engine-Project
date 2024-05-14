import csv
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import pandas as pd


def read_queries(queries_file):
    queries = {}
    with open(queries_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            queries[row[0]] = row[1]
    return queries


def read_qrels(qrels_file):
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split(',')
            qrels[query_id][doc_id] = int(relevance)
    return qrels


def get_results():
    pass



def calculate_AP(query_id, query_text, qrels):
    relevant_docs = qrels.get(query_id, {})
    ordered_results = get_results(query_text)
    total_relevant = sum(relevant_docs.values())
    if total_relevant == 0:
        return 0
    precision_sum = 0
    relevant_retrieved = 0
    for i, result in enumerate(ordered_results, start=1):
        doc_id = str(result['_id'])
        if doc_id in relevant_docs:
            relevant_retrieved += 1
            precision_sum += relevant_retrieved / i
    return precision_sum / total_relevant


def evaluate_retrieval_system(queries_file, qrels_file):

    queries = read_queries(queries_file)
    qrels = read_qrels(qrels_file)
    APs = []
    for query_id, query_text in queries.items():
        AP = calculate_AP(query_id, query_text, qrels)
        APs.append(AP)
    MAP = sum(APs) / len(APs)
    return MAP


# Example usage
if __name__ == "__main__":

    queries_file = 'queries.csv'
    qrels_file = 'qrels.csv'


    MAP = evaluate_retrieval_system(queries_file, qrels_file)
    print(f"Mean Average Precision (MAP): {MAP}")
