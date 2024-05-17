from dataset.WikipediaReader import WikipediaReader
import os

from models.tf_idf.base_tf_idf_model import BaseTFIDFModel
from text_processors.wikipedia_text_processor import WikipediaTextProcessor


class WikipediaTFIDFModel(BaseTFIDFModel):

    def __init__(self):
        super(WikipediaTFIDFModel, self).__init__(
            model_name='wikipedia',
            dataset_reader=WikipediaReader(file_path=os.environ.get('WIKIPEDIA_DATASET_PATH')),
            text_processor=WikipediaTextProcessor(),
        )
