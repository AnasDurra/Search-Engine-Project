from pymongo import MongoClient
import os


class MongoDBConnection:
    __instance = None

    @staticmethod
    def get_instance():
        """
        Static method to get the singleton instance of the MongoDBConnection class.
        """
        if MongoDBConnection.__instance is None:
            MongoDBConnection()
        return MongoDBConnection.__instance

    def __init__(self):
        """
        Private constructor to initialize the MongoDB connection.
        """
        if MongoDBConnection.__instance is not None:
            raise Exception("MongoDBConnection is a singleton class. Use get_instance() method to get the instance.")
        else:
            # MongoDB's connection details
            db_url: str = os.environ['MONGODB_URL']
            self.db_name: str = os.environ['MONGODB_DBNAME']
            self.client = MongoClient(db_url)
            MongoDBConnection.__instance = self

    def get_collection(self, collection_name):
        """
        Method to get a specific collection instance from the global database.
        """
        db = self.client[self.db_name]
        return db.get_collection(collection_name)

    def create_collection(self, collection_name):
        """
        Method to create a new collection in the global database.
        """
        db = self.client[self.db_name]
        return db.create_collection(collection_name)

    def collection_exists(self, collection_name) -> bool:
        """
        Method to check if a specific collection exists in the global database.
        """
        db = self.client[self.db_name]
        return collection_name in db.list_collection_names()
