import math
from typing import Optional, List

import numpy as np
import pandas as pd
from pymongo.collection import Collection
from sklearn.feature_extraction.text import TfidfVectorizer
from common.constants import Locations
from common.file_utilities import FileUtilities
from database.mongo_helper import MongoDBConnection
from dataset.dataset_reader import DatasetReader
from text_processors.base_text_processor import BaseTextProcessor


class BaseTFIDFModel:

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
        self.db_connection = MongoDBConnection.get_instance()

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

        # second, check if the dataset is stored in the database or not.
        if not self.db_connection.collection_exists(self.model_name):
            self.store_dataset_in_db(dataset=dataset)

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

    def store_dataset_in_db(self, dataset: dict) -> None:
        # create a new collection
        self.db_connection.create_collection(self.model_name)

        # get the created collection
        collection: Collection = self.db_connection.get_collection(self.model_name)

        # insert documents as chunks (5000 in each)
        chunk_size = 5000
        dataset_items = list(dataset.items())
        num_chunks = math.ceil(len(dataset_items) / chunk_size)

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(dataset_items))
            chunk = dataset_items[start_idx:end_idx]

            documents = [
                {
                    "doc_id": key,
                    "doc_content": value,
                    "index": idx + 1 + start_idx
                }
                for idx, (key, value) in enumerate(chunk)
            ]

            collection.insert_many(documents)
