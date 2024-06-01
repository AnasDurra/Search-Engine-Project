import csv
from abc import ABC
from collections import defaultdict
import os

from overrides import overrides
import pandas as pd
from dataset.dataset_reader import DatasetReader


class WikipediaReader(DatasetReader, ABC):

    @overrides
    def load_as_dict(self) -> dict:
        key_value_pairs = {}

        df = pd.read_csv(self.file_path)

        for index, row in df.iterrows():
            key = str(row['id_right'])
            value = str(row['text_right'])

            key_value_pairs[key] = value

        return key_value_pairs

    @overrides
    def read_queries(self) -> dict:
        queries_path = os.environ.get('WIKIPEDIA_QUERIES_PATH', '../data/wikipedia/queries.csv')
        queries = {}
        with open(queries_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                queries[row[0]] = row[1]
        return queries

    @overrides
    def read_qrels(self) -> defaultdict:
        qrels_path = os.environ.get('WIKIPEDIA_QRELS_PATH', '../data/wikipedia/qrels')
        qrels = defaultdict(dict)

        with open(qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                query_id, _, doc_id, relevance = parts
                qrels[query_id][doc_id] = int(relevance)

        return qrels


if __name__ == "__main__":

    file_path = "../data/wikipedia/wikipedia-en.csv"
    reader = WikipediaReader(file_path)
    wikipedia_data = reader.load_as_dict()

    print("First 5 key-value pairs:")
    for key, value in list(wikipedia_data.items())[:5]:
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
