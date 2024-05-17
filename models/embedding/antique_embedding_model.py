import os

from common.constants import Locations
from dataset.antique_reader import AntiqueReader
from models.embedding.base_embedding_model import BaseEmbeddingModel
from text_processors.antique_text_processor import AntiqueTextProcessor


class AntiqueEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        model_name: str = 'antique'
        antique_dataset_file_name: str = f'{model_name}.txt'
        super().__init__(
            dataset_reader=AntiqueReader(Locations.generate_dataset_path(antique_dataset_file_name)),
            text_processor=AntiqueTextProcessor(),
            model_name=model_name
        )
