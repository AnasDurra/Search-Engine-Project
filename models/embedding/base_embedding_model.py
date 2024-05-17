from typing import Optional

from chromadb.api.models.Collection import Collection
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from common.constants import Locations
from common.file_utilities import FileUtilities
from database.chroma_helper import ChromaHelper
from dataset.dataset_reader import DatasetReader
from text_processors.base_text_processor import BaseTextProcessor


class BaseEmbeddingModel:

    def __init__(
            self,
            dataset_reader: DatasetReader,
            text_processor: BaseTextProcessor,
            model_name: str,
    ):
        self.vector_db_helper = ChromaHelper.get_instance()
        self.dataset_reader = dataset_reader
        self.text_processor = text_processor
        self.model_name = model_name

        # Initialize Doc2Vec model
        # self.model = Doc2Vec(
        #     vector_size=300,
        #     window=5,
        #     min_count=2,
        #     workers=5,
        #     epochs=15
        # )
        self.model: Doc2Vec = FileUtilities.load_file(
            file_path=Locations.generate_embeddings_model_path(model_name)
        )

    def train(self) -> None:
        # load the dataset
        dataset: dict = self.dataset_reader.load_as_dict()

        # # extract the list of data
        # documents: list = list(dataset.values())
        #
        # # # Preprocess the documents and create TaggedDocument objects
        # # tagged_data = [
        # #     TaggedDocument(
        # #         words=self.text_processor.process(doc),
        # #         tags=[str(i)]
        # #     ) for i, doc in enumerate(documents)
        # # ]
        # #
        # # # Build vocabulary
        # # self.model.build_vocab(tagged_data)
        # #
        # # # Train the model
        # # self.model.train(
        # #     tagged_data,
        # #     total_examples=self.model.corpus_count,
        # #     epochs=self.model.epochs
        # # )
        # #
        # # # Save the model
        # # self.save_model()

        # Save the embeddings
        self.save_embeddings(dataset)

    def save_model(self) -> None:
        FileUtilities.save_file(
            file_path=Locations.generate_embeddings_model_path(self.model_name),
            content=self.model
        )

    def __embedding_function(self, text: str):
        processed_text = self.text_processor.process(text)
        return self.model.infer_vector(processed_text)

    def save_embeddings(self, dataset: dict, batch_size: int = 5000):
        collection: Collection = self.vector_db_helper.get_or_create_collection(
            collection_name=self.model_name,
        )

        documents: list = list(dataset.values())
        ids: list = list(dataset.keys())

        # Split documents and IDs into batches
        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # Infer embeddings for the current batch
            batch_embeddings = [
                self.model.infer_vector(self.text_processor.process(doc)).tolist()
                for doc in batch_documents
            ]

            # Add the batch to the collection
            collection.add(
                documents=batch_documents,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
