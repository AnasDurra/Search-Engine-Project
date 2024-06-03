import time

import joblib
import pandas as pd
from sklearn.cluster import KMeans
import os

from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

from common.constants import Locations
from common.file_utilities import FileUtilities
from database.mongo_helper import MongoDBConnection

from dotenv import load_dotenv

load_dotenv()


class DocsClustering:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Load the TF-IDF Matrix
        matrix_path: str = Locations.generate_matrix_path(model_name)
        self.matrix = FileUtilities.load_file(matrix_path)

        # Load the model
        model_path: str = Locations.generate_model_path(model_name)
        self.model: TfidfVectorizer = FileUtilities.load_file(model_path)

        # Database client
        self.db_collection = MongoDBConnection.get_instance().get_collection(model_name)
        self.documents = pd.DataFrame([doc for doc in self.db_collection.find()])
        print(len(self.documents))

        self.pkl_files_path = os.environ.get('CLUSTERS_PATH')

    def __choose_number_of_clusters(self):
        print("start choosing")
        wcss = []
        for i in range(1, 10):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.matrix)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, 50), wcss)
        plt.title('Elbow Method using MiniBatchKMeans')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    def __print_silhouette_scores(self, labels):
        silhouette_vals = silhouette_samples(self.matrix, labels)
        for i, silhouette in enumerate(silhouette_vals):
            print(f"Document {i}: Silhouette Score = {silhouette}")
        avg_score = np.mean(silhouette_vals)
        print(f"Average Silhouette Score = {avg_score}")

    def __cluster(self, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(self.matrix)
        cluster_labels = kmeans.labels_

        # create a DataFrame to hold the TF-IDF vectors and their corresponding cluster labels
        df = pd.DataFrame.sparse.from_spmatrix(self.matrix)
        df['cluster'] = cluster_labels
        print(df)

        output_dir = f"{self.pkl_files_path}/{self.model_name}_clusters"
        os.makedirs(output_dir, exist_ok=True)

        # Measure time for the original operation
        start_time = time.time()
        start_original = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print(start_original)

        # Save each cluster
        for cluster_num in range(k):
            print(f"Cluster {cluster_num}")
            cluster_data = df[df['cluster'] == cluster_num].drop(columns='cluster')
            cluster_indices = df[df['cluster'] == cluster_num].index.values + 1

            sparse_matrix = csr_matrix(cluster_data.sparse.to_coo())

            cluster_file = os.path.join(output_dir, f'cluster{cluster_num}.pkl')

            with open(cluster_file, 'wb') as f:
                print((sparse_matrix,cluster_indices))
                joblib.dump((sparse_matrix, cluster_indices), f)

        end_time = time.time()
        end_original = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        print(end_original)

        self.documents['cluster'] = cluster_labels

        for idx, row in self.documents.iterrows():
            self.db_collection.update_one({'_id': row['_id']}, {'$set': {'cluster': int(row['cluster'])}})

        self.__plot_clusters(cluster_labels)

    def __plot_clusters(self, labels):
        svd = TruncatedSVD(n_components=2)
        tfidf_matrix_2d = svd.fit_transform(self.matrix)
        scaler = StandardScaler()
        tfidf_matrix_2d_scaled = scaler.fit_transform(tfidf_matrix_2d)
        offset = 50
        scale_factor = 10
        tfidf_matrix_2d_scaled = tfidf_matrix_2d_scaled * scale_factor + offset
        plt.figure(figsize=(10, 8))
        plt.scatter(tfidf_matrix_2d_scaled[:, 0], tfidf_matrix_2d_scaled[:, 1], c=labels, cmap='viridis')
        plt.title('Clustering result')
        plt.colorbar()
        plt.show()

    def perform_clustering(self):
        # self.__choose_number_of_clusters()
        k = int(input("Enter the number of clusters based on the Elbow method plot: "))
        self.__cluster(k)
