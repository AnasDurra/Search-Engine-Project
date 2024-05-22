import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.constants import Locations
from common.file_utilities import FileUtilities
from database.mongo_helper import MongoDBConnection


class QueryMatcher:
    def __init__(self, model_name: str):
        # Load the TF-IDF Matrix
        matrix_path: str = Locations.generate_matrix_path(model_name)
        self.matrix = FileUtilities.load_file(matrix_path)

        # Load the model
        model_path: str = Locations.generate_model_path(model_name)
        self.model: TfidfVectorizer = FileUtilities.load_file(model_path)

        # Variable that affects engine accuracy
        self.threshold = float(os.environ.get('SIMILARITY_THRESHOLD', 0.5))

        # Database client
        self.db_collection = MongoDBConnection.get_instance().get_collection(model_name)

    def __vectorize_query(self, query: str):
        return self.model.transform([query])

    def match(self, query: str, n):
        print(f"Query: {query}")
        # Vectorize the query
        query_vector = self.__vectorize_query(query)
        # print(query_vector)

        # Calculate cosine similarity between query vector and document vectors
        cos_similarities = cosine_similarity(self.matrix, query_vector)

        # Sort the cosine similarities in descending order
        sorted_indices = np.argsort(cos_similarities, axis=0)[::-1]

        # Get all matching documents indices based on threshold
        matching_docs_indices = []
        for i in sorted_indices:
            if cos_similarities[i].item() >= self.threshold:
                matching_docs_indices.append(i.item()+1)

        # Get the documents associated with the sorted cosine similarities
        matching_results = list(self.db_collection.find({"index": {"$in": matching_docs_indices}}))

        return sorted(
            matching_results,
            key=lambda x: matching_docs_indices.index(x['index']),
            reverse=True
        )
