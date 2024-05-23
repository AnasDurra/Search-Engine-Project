from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from common.constants import Locations
from common.file_utilities import FileUtilities
from database.mongo_helper import MongoDBConnection
from collections import defaultdict

class DocsClustering:
    def __init__(self, model_name: str):
        # Load the TF-IDF Matrix
        matrix_path: str = Locations.generate_matrix_path(model_name)
        self.matrix = FileUtilities.load_file(matrix_path)

        # Load the model
        model_path: str = Locations.generate_model_path(model_name)
        self.model: TfidfVectorizer = FileUtilities.load_file(model_path)

        # Database client
        # self.db_collection = MongoDBConnection.get_instance().get_collection(model_name)

    def __vectorize_query(self, query: str):
        return self.model.transform([query])

    def __choose_number_of_clusters(self):
        print("start choosing")
        sum_of_squared_distances = []
        K = range(1, 10)
        for k in K:
            km = KMeans(n_clusters=k, random_state=0, n_init='auto')
            km = km.fit(self.matrix)
            sum_of_squared_distances.append(km.inertia_)

        plt.plot(K, sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def __cluster(self, k=5):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(self.matrix)
        labels = kmeans.labels_
        self.__print_clusters(labels)
        print("start evaluation")
        score = silhouette_score(self.matrix, labels)
        print(f'Silhouette Score: {score}')
        self.__visualize_clusters(kmeans, labels)

    def __print_clusters(self, labels):
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        # for cluster_id, document_indices in clusters.items():
        #     print(f"Cluster {cluster_id}:")
        #     for doc_index in document_indices:
        #         print(f" Document {doc_index}")  # Replace with actual document titles or snippets

    def __visualize_clusters(self, kmeans, labels):
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.matrix)
        reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels)
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=200, c='r')
        plt.show()

    def perform_clustering(self):
        self.__choose_number_of_clusters()
        k = int(input("Enter the number of clusters based on the Elbow method plot: "))
        self.__cluster(k)
