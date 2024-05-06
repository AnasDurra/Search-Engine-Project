from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer

from common.constants import Locations
from common.file_utilities import FileUtilities
from dataset.dataset_reader import DatasetReader
from text_processors.base_text_processor import BaseTextProcessor


class BaseEmbeddingModel:

    # initialize default variables
    def __init__(
            self,
            dataset_reader: DatasetReader,
            text_processor: BaseTextProcessor,
            model_name: str,
    ):
        self.model_name = model_name
        self.dataset_reader = dataset_reader
        self.text_processor = text_processor

        # Ensure that required parameters are provided
        if not all([self.model_name, self.dataset_reader, self.text_processor]):
            raise ValueError("Required parameters are missing.")

        # TODO: CUSTOMIZE MODEL PREFERENCES AS YOU NEED IN SUBCLASS
        self.vectorizer = TfidfVectorizer(
            tokenizer=text_processor.process,
            token_pattern=None,
            lowercase=True,
            max_df=0.5,
            min_df=2,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )

    def train(self, number_of_docs: Optional[int] = None) -> None:
        # first, load the dataset
        dataset: dict = self.dataset_reader.load_as_dict()

        # extract the list of data based on the number_of_docs argument
        if number_of_docs is not None:
            documents: list = list(dataset.values())[:number_of_docs]
        else:
            documents: list = list(dataset.values())

        # pass the training documents to the model
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # store the resulted TF-IDF Matrix
        FileUtilities.save_file(Locations.generate_matrix_path(self.model_name), tfidf_matrix)

        # store the vectorizer model (in project folder called engines)
        FileUtilities.save_file(Locations.generate_model_path(self.model_name), self.vectorizer)
