import os

from common.constants import Locations
from dataset.antique_reader import AntiqueReader
from models.embedding.base_embedding_model import BaseEmbeddingModel
from text_processors.antique_text_processor import AntiqueTextProcessor


class AntiqueEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        model_name: str = 'antique'
        antique_dataset_file_path: str = os.environ.get('ANTIQUE_DATASET_PATH')
        antique_reader: AntiqueReader = AntiqueReader(antique_dataset_file_path)
        super().__init__(
            dataset_reader=antique_reader,
            text_processor=AntiqueTextProcessor(),
            model_name=model_name
        )
        self.vector_size = int(os.environ.get('ANTIQUE_VECTOR_SIZE', 500))
        self.epochs = int(os.environ.get('ANTIQUE_EPOCHS', 50))
