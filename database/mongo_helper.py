from pymongo import MongoClient
import os


class MongoDBConnection:
    __instance = None

    @staticmethod
    def get_instance() -> 'MongoDBConnection':
        if MongoDBConnection.__instance is None:
            MongoDBConnection()
        return MongoDBConnection.__instance

    def __init__(self):
        if MongoDBConnection.__instance is not None:
            raise Exception("MongoDBConnection is a singleton class. Use get_instance() method to get the instance.")
        else:
            # MongoDB's connection details
            db_url: str = os.environ['MONGODB_URL']
            self.db_name: str = os.environ['MONGODB_DBNAME']
            self.client = MongoClient(db_url)
            MongoDBConnection.__instance = self

    def get_collection(self, collection_name):
        db = self.client[self.db_name]
        return db.get_collection(collection_name)

    def create_collection(self, collection_name):
        db = self.client[self.db_name]
        return db.create_collection(collection_name)

    def collection_exists(self, collection_name) -> bool:
        db = self.client[self.db_name]
        return collection_name in db.list_collection_names()
