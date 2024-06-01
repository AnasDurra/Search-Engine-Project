import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from common.constants import Locations
from common.file_utilities import FileUtilities
from database.mongo_helper import MongoDBConnection


class DocumentSearch:
    def __init__(self, db_collection_name: str, model_path: str):
        self.db_collection = MongoDBConnection.get_instance().get_collection(db_collection_name)
        self.model_path = Locations.generate_model_path(model_path)
        self.tfidf_vectorizer = FileUtilities.load_file(self.model_path)

    def get_document_vector(self, doc_content: str):
        return self.tfidf_vectorizer.transform([doc_content])

    def find_similar_documents(self, document: dict, threshold: float = 0.1):
        document_vector = self.get_document_vector(document["doc_content"])
        cluster_num = document["cluster"]

        pkl_file_path = f"clusters/antique_clusters/cluster{cluster_num}.pkl"

        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                sparse_matrix, cluster_indices = joblib.load(f)
                cos_similarities = cosine_similarity(sparse_matrix, document_vector)
                combined_data = [(cos_similarities[i][0], cluster_indices[i]) for i in range(len(cos_similarities))]

                sorted_indices = sorted(combined_data, key=lambda x: x[0], reverse=True)

                matching_docs_indices = [index for similarity, index in sorted_indices if similarity >= threshold]

                matching_docs_indices = [int(idx) for idx in matching_docs_indices]
                matching_results = list(self.db_collection.find({"index": {"$in": matching_docs_indices}}))

                return sorted(matching_results, key=lambda x: matching_docs_indices.index(x['index']))
        else:
            raise FileNotFoundError(f"Pickle file '{pkl_file_path}' does not exist.")

    def query(self, document: dict):
        try:
            return self.find_similar_documents(document)
        except FileNotFoundError as e:
            return str(e)


# Example usage
if __name__ == "__main__":
    search = DocumentSearch("antique", "antique")
    document = {
        "_id": "665af6f91bdb60b7be02db86",
        "doc_id": "2474377_0",
        "doc_content": "Iraq and Iran",
        "cluster": 1,
        "index": 509
    }
    results = search.query(document)
    print(results)
