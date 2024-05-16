import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.constants import Locations
from common.file_utilities import FileUtilities
from database.mongo_helper import MongoDBConnection


class QueryMatcher:
    def __init__(self, model_name: str):
        # load the TF-IDF Matrix
        self.matrix = FileUtilities.load_file(Locations.generate_stored_matrix_path(model_name))

        # load the model
        self.model: TfidfVectorizer = FileUtilities.load_file(Locations.generate_stored_model_path(model_name))
        # variable that affect engine accuracy
        self.threshold = float(os.environ.get('SIMILARITY_THRESHOLD', 0.5))

        # database client
        self.db_collection = MongoDBConnection.get_instance().get_collection(model_name)

    def __vectorize_query(self, query: str):
        return self.model.transform([query])

    def match(self, query: str, top: int = 10):
        print(f"Query: {query}")
        # vectorize the query.
        query_vector = self.__vectorize_query(query)

        # Calculate cosine similarity between query vector and document vectors
        cos_similarities = cosine_similarity(self.matrix, query_vector)

        # Sort the cosine similarities in descending order
        sorted_indices = np.argsort(cos_similarities, axis=0)[::-1]

        # Get top k result based on input
        matching_docs_indices = []
        for i in sorted_indices:

            if len(matching_docs_indices) > top:
                break

            if cos_similarities[i] >= self.threshold:
                matching_docs_indices.append(i.item())

        # Get the documents associated with the sorted cosine similarities
        matching_results = list(self.db_collection.find({"index": {"$in": matching_docs_indices}}))
        return sorted(
            matching_results,
            key=lambda x: matching_docs_indices.index(x['index']),
            reverse=True
        )
