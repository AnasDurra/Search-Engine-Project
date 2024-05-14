from common.constants import DatasetNames
from dataset.antique_reader import AntiqueReader
from models.tf_idf.base_tf_idf_model import BaseTFIDFModel
from text_processors.antique_text_processor import AntiqueTextProcessor
import os


class AntiqueTFIDFModel(BaseTFIDFModel):

    def __init__(self):
        super(AntiqueTFIDFModel, self).__init__(
            model_name=DatasetNames.ANTIQUE,
            dataset_reader=AntiqueReader(file_path=os.environ.get('ANTIQUE_DATASET_PATH', '../../data/antique.txt')),
            text_processor=AntiqueTextProcessor(),
        )
