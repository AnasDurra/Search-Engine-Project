import os

import chromadb
from chromadb.api.models.Collection import Collection

from common.constants import Locations


class ChromaHelper:
    __instance = None

    @staticmethod
    def get_instance() -> 'ChromaHelper':
        """
        Static method to get the singleton instance of the ChromaHelper class.
        """
        if ChromaHelper.__instance is None:
            ChromaHelper()
        return ChromaHelper.__instance

    def __init__(self):
        """
        Private constructor to initialize the Chroma connection.
        """
        if ChromaHelper.__instance is not None:
            raise Exception("ChromaHelper is a singleton class. Use get_instance() method to get the instance.")
        else:

            # Chroma's connection details
            self.client = chromadb.PersistentClient(Locations.generate_chroma_db_path())
            ChromaHelper.__instance = self

    def get_or_create_collection(self, collection_name: str) -> Collection:
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
