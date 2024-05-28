import os
from typing import List

import numpy as np
from gensim.models import Doc2Vec, Word2Vec
from numpy import ndarray

from common.constants import Locations
from common.file_utilities import FileUtilities
from database.chroma_helper import ChromaHelper
from text_processors.base_text_processor import BaseTextProcessor


class BaseEmbeddingMatcher:

    def __init__(self, model_name: str, text_processor: BaseTextProcessor):
        self.vector_collection = ChromaHelper.get_instance().get_or_create_collection(model_name)
        self.vector_size = int(os.environ.get("VECTOR_SIZE", 500))
        self.model: Word2Vec = self.__load_model(model_name)
        self.text_processor = text_processor
        self.model_name = model_name

    def match(self, text: str, top: int = 10):
        # preprocess the query
        processed_query: List[str] = self.text_processor.process_query(text)

        # create embeddings
        query_embeddings: List = self.vectorize_query(processed_query).tolist()

        # query the vector db for similar docs.
        result = self.vector_collection.query(
            query_embeddings=query_embeddings,
            n_results=top,
        )

        # Transforming the output to the desired format
        transformed_results = []
        ids = result.get('ids', [[]])[0]
        documents = result.get('documents', [[]])[0]
        distances = result.get('distances', [[]])[0]

        for doc_id, doc_content, doc_similarity in zip(ids, documents, distances):
            transformed_results.append({
                'doc_id': doc_id,
                'doc_content': doc_content,
                'similarity': doc_similarity,
            })

        return transformed_results

    def vectorize_query(self, query_words: list[str]) -> ndarray:

        # Collect query vectors for the query string
        query_vectors = [self.model.wv[word] for word in query_words if word in self.model.wv]

        if query_vectors:
            # Calculate the mean vector if there are valid word vectors
            query_vec = np.mean(query_vectors, axis=0)
        else:
            # Use an initial vector if there are no valid word vectors
            query_vec = np.zeros(self.vector_size)

        return query_vec

    @staticmethod
    def __load_model(model_name: str):
        return FileUtilities.load_file(
            file_path=Locations.generate_embeddings_model_path(model_name)
        )
