from common.constants import DatasetNames
from dataset.antique_reader import AntiqueReader
from models.base_embedding_model import BaseEmbeddingModel
from text_processors.antique_text_processor import AntiqueTextProcessor
import os


class AntiqueEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        super(AntiqueEmbeddingModel, self).__init__(
            model_name=DatasetNames.ANTIQUE,
            dataset_reader=AntiqueReader(file_path=os.environ.get('ANTIQUE_DATASET_PATH', '../data/antique.txt')),
            text_processor=AntiqueTextProcessor(),
        )
