import os

import numpy as np
from chromadb.api.models.Collection import Collection
from gensim.models import Word2Vec

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
        # load env vars
        self.vector_size = int(os.environ.get('VECTOR_SIZE', 500))
        self.skip_gram = int(os.environ.get('SKIP_GRAM', 1))
        self.workers = int(os.environ.get('WORKERS', 4))
        self.epochs = int(os.environ.get('EPOCHS', 50))

        # initialize data
        self.model: Word2Vec | None = None  # initialize model to none.
        self.vector_db_helper = ChromaHelper.get_instance()
        self.dataset_reader = dataset_reader
        self.text_processor = text_processor
        self.model_name = model_name

    def __prepare_docs(self) -> list[dict]:
        # processed data
        docs = []

        # load the data
        documents: dict = self.dataset_reader.load_as_dict()

        # pre-process data
        for doc_id, doc_content in documents.items():
            # apply text processing to document's content
            processed_doc = self.text_processor.process(doc_content)
            docs.append({
                'doc_id': doc_id,
                'doc_content': doc_content,
                'processed_doc': processed_doc
            })

        return docs

    def train(self) -> None:

        # Load the pre-processed documents
        documents: list[dict] = self.__prepare_docs()

        # Extract the processed documents for training the Word2Vec model
        tokenized_docs = [doc['processed_doc'] for doc in documents]

        # Initialize Word2Vec model
        self.model = Word2Vec(
            min_count=1,
            vector_size=self.vector_size,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.skip_gram,
        )

        
        # prepare the model vocabulary
        self.model.build_vocab(tokenized_docs)

        # train word vectors
        self.model.train(tokenized_docs, total_examples=self.model.corpus_count, epochs=self.epochs)

        # Save the model
        self.save_model()

        # Save the embeddings
        self.save_embeddings(documents)

    def save_model(self) -> None:
        FileUtilities.save_file(
            file_path=Locations.generate_embeddings_model_path(self.model_name),
            content=self.model
        )

    def save_embeddings(self, docs_data):
        batch_size = 5000

        # Split docs_data into batches of 5000
        for i in range(0, len(docs_data), batch_size):
            batch = docs_data[i:i + batch_size]
            batch_vectors = []

            for doc in batch:
                # Collect word vectors for the current document
                doc_vectors = []
                for word in doc['processed_doc']:
                    if word in self.model.wv:
                        try:
                            doc_vectors.append(self.model.wv[word])
                        except KeyError:
                            doc_vectors.append(np.random.rand(self.vector_size))

                if doc_vectors:
                    # Calculate the mean vector if there are valid word vectors
                    doc_vec = np.mean(doc_vectors, axis=0)
                else:
                    # Use an initial vector if there are no valid word vectors
                    doc_vec = np.zeros(self.vector_size)

                batch_vectors.append(doc_vec)

            # Prepare the batch payload
            for j, doc in enumerate(batch):
                doc['embedding_vector'] = batch_vectors[j]

            # Add batch to database
            self.__add_vectors_to_database(batch)

    def __add_vectors_to_database(self, documents: list[dict]):

        # get reference to the db collection
        collection: Collection = self.vector_db_helper.get_or_create_collection(
            collection_name=self.model_name
        )

        # extract data
        contents: list[str] = [doc['doc_content'] for doc in documents]
        ids: list[str] = [doc['doc_id'] for doc in documents]
        embeddings: list[list] = [doc['embedding_vector'].tolist() for doc in documents]

        # insert data to db
        collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
        )

    # def save_embeddings(self, dataset: dict, batch_size: int = 5000):
#     collection: Collection = self.vector_db_helper.get_or_create_collection(
#         collection_name=self.model_name,
#     )
#
#     documents: list = list(dataset.values())
#     ids: list = list(dataset.keys())
#
#     # Split documents and IDs into batches
#     for i in range(0, len(documents), batch_size):
#         batch_documents = documents[i:i + batch_size]
#         batch_ids = ids[i:i + batch_size]
#
#         # Infer embeddings for the current batch
#         batch_embeddings = [
#             self.model.infer_vector(self.text_processor.process(doc)).tolist()
#             for doc in batch_documents
#         ]
#
#         # Add the batch to the collection
#         collection.add(
#             documents=batch_documents,
#             ids=batch_ids,
#             embeddings=batch_embeddings
#         )
