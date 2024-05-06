from dataset.antique_reader import AntiqueReader
from models.base_embedding_model import BaseEmbeddingModel
from text_processors.antique_text_processor import AntiqueTextProcessor


class AntiqueEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        super(AntiqueEmbeddingModel, self).__init__(
            model_name='antique',
            dataset_reader=AntiqueReader(file_path='../data/test.txt'),
            text_processor=AntiqueTextProcessor(),
        )
