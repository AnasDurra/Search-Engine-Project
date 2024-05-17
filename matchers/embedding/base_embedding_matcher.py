from typing import List

from gensim.models import Doc2Vec

from common.constants import Locations
from common.file_utilities import FileUtilities
from database.chroma_helper import ChromaHelper
from text_processors.base_text_processor import BaseTextProcessor


class BaseEmbeddingMatcher:

    def __init__(self, model_name: str, text_processor: BaseTextProcessor):
        self.vector_collection = ChromaHelper.get_instance().get_or_create_collection(model_name)
        self.model: Doc2Vec = self.__load_model(model_name)
        self.text_processor = text_processor
        self.model_name = model_name

    def match(self, text: str, top: int = 10):
        # preprocess the query
        processed_query: List[str] = self.text_processor.process(text)

        # create embeddings
        query_embeddings: List = self.model.infer_vector(processed_query).tolist()

        # query the vector db for similar docs.
        return self.vector_collection.query(
            query_embeddings=query_embeddings,
            n_results=top,
        )

    @staticmethod
    def __load_model(model_name: str):
        return FileUtilities.load_file(
            file_path=Locations.generate_embeddings_model_path(model_name)
        )