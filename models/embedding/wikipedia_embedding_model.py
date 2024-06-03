import os

from common.constants import Locations
from dataset.WikipediaReader import WikipediaReader
from models.embedding.base_embedding_model import BaseEmbeddingModel
from text_processors.wikipedia_text_processor import WikipediaTextProcessor


class WikipediaEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        model_name: str = 'wikipedia'
        wikipedia_dataset_file_path: str = os.environ.get('WIKIPEDIA_DATASET_PATH')
        wikipedia_reader: WikipediaReader = WikipediaReader(wikipedia_dataset_file_path)
        super().__init__(
            dataset_reader=wikipedia_reader,
            text_processor=WikipediaTextProcessor(),
            model_name=model_name
        )
        self.vector_size = int(os.environ.get('WIKIPEDIA_VECTOR_SIZE', 500))
        self.epochs = int(os.environ.get('WIKIPEDIA_EPOCHS', 50))
