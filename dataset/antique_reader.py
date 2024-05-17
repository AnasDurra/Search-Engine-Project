import os
from collections import defaultdict
import re
from overrides import overrides
from dataset.dataset_reader import DatasetReader


class AntiqueReader(DatasetReader):
    @overrides
    def load_as_dict(self) -> dict:
        key_value_pairs = {}
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                key, value = line.strip().split('\t')
                key_value_pairs[key] = value
        return key_value_pairs

    @overrides
    def read_queries(self) -> dict:
        queries_path = os.environ.get('ANTIQUE_QUERIES_PATH', '../data/antique/queries.txt')
        queries = {}

        # Read the queries file
        with open(queries_path, 'r') as f:
            for line in f:
                query_id, query_text = line.strip().split('\t')  # Assuming tab-separated values
                queries[query_id] = query_text

        return queries

    @overrides
    def read_qrels(self) -> defaultdict:
        qrels_path = os.environ.get('ANTIQUE_QRELS_PATH', '../data/antique/qrels')
        qrels = defaultdict(dict)

        # Read the qrels file
        with open(qrels_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = re.split(r'\s+', line.strip())

                query_id, _, doc_id, relevance = parts
                qrels[query_id][doc_id] = int(relevance)

        return qrels


if __name__ == "__main__":

    file_path = "../data/antique/antique.txt"
    reader = AntiqueReader(file_path)
    antique_date = reader.load_as_dict()

    print("First 5 key-value pairs:")
    for key, value in list(antique_date.items())[:5]:
        print(f"Key: {key}, Value: {value}")

    queries = reader.read_queries()
    print("\nFirst 5 queries:")
    for key, value in list(queries.items())[:5]:
        print(f"Query ID: {key}, Query Text: {value}")

    qrels = reader.read_qrels()
    print("\nFirst 5 Qrels:")
    for query_id, docs in list(qrels.items())[:5]:
        print(f"Query ID: {query_id}")
        for doc_id, relevance in docs.items():
            print(f"  Doc ID: {doc_id}, Relevance: {relevance}")
