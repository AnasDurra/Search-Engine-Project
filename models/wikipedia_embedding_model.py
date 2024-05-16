from dataset.WikipediaReader import WikipediaReader
from models.base_embedding_model import BaseEmbeddingModel
import os

from text_processors.wikipedia_text_processor import WikipediaTextProcessor


class WikipediaEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        super(WikipediaEmbeddingModel, self).__init__(
            model_name='wikipedia',
            dataset_reader=WikipediaReader(file_path=os.environ.get('WIKIPEDIA_DATASET_PATH', '../data/wikipedia'
                                                                                              '/wikipedia-en.csv')),
            text_processor=WikipediaTextProcessor(),
        )
