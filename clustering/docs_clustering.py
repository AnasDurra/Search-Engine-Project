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
        self.db_collection = MongoDBConnection.get_instance().get_collection(model_name)
        self.documents = pd.DataFrame([doc for doc in self.db_collection.find()])
        print(len(self.documents))

    def __choose_number_of_clusters(self):
        print("start choosing")
        wcss = []
        for i in range(1, 50):
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
        kmeans = KMeans(n_clusters=k, n_init=25, random_state=12345)
        kmeans.fit(self.matrix)

        cluster_labels = kmeans.labels_
        # create a DataFrame to hold the TF-IDF vectors and their corresponding cluster labels
        df = pd.DataFrame(self.matrix.toarray())
        df['cluster'] = cluster_labels
        output_dir = 'clusters/antique_clusters'
        os.makedirs(output_dir, exist_ok=True)

        # Save each cluster
        for cluster_num in range(k):
            print(f"Cluster {cluster_num}")
            cluster_data = df[df['cluster'] == cluster_num].drop(columns='cluster')
            cluster_indices = df[df['cluster'] == cluster_num].index.values + 1
            # Convert the DataFrame back to a sparse matrix
            sparse_matrix = csr_matrix(cluster_data.values)
            cluster_file = os.path.join(output_dir, f'cluster{cluster_num}.pkl')
            # Save the sparse matrix and indices in .pkl format
            with open(cluster_file, 'wb') as f:
                joblib.dump((sparse_matrix, cluster_indices), f)

        self.documents['cluster'] = kmeans.labels_

        # Update MongoDB with the new cluster labels
        for idx, row in self.documents.iterrows():
            self.db_collection.update_one({'_id': row['_id']}, {'$set': {'cluster': int(row['cluster'])}})

        # # output the result to a text files
        # clusters = self.documents.groupby('cluster')
        # for cluster in clusters.groups:
        #     f = open(f'clusters/antique_clusters/cluster{cluster}.csv', 'w')
        #     data = clusters.get_group(cluster)[['doc_id', 'doc_content', 'index']]
        #     f.write(data.to_csv(index_label='_id'))
        #     f.close()
        #
        # print("Clusters Centroids : \n ")
        # order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        # terms = self.model.get_feature_names_out()
        #
        # for i in range(k):
        #     print("Cluster %d:" % i)
        #     for j in order_centroids[i, :10]:  # print out 10 feature terms of each cluster
        #         print(' %s' % terms[j])
        #     print('------------')
        # labels = kmeans.labels_

        # self.__print_clusters(labels)
        # score = silhouette_score(self.matrix, labels)
        # print(f'Silhouette Score: {score}')
        # self.__visualize_clusters(kmeans, cluster_labels)
        self.__plot_clusters(cluster_labels)
        # self.__print_silhouette_scores(labels)

    def __visualize_clusters(self, kmeans, labels):
        # Predicting the clusters
        labels = kmeans.labels_
        # Getting the cluster centers
        C = kmeans.cluster_centers_

        # transform n variables to 2 principal components to plot
        pca = PCA(n_components=2)
        pca_fit = pca.fit(self.matrix)
        principalComponents = pca.fit_transform(self.matrix)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'lime', 'orange', 'coral', 'brown', 'peru', 'khaki', 'tan']
        centroidColor = []
        for item in range(3):
            centroidColor.append(colors[item])

        dataPointColor = []
        for row in labels:
            dataPointColor.append(colors[row])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
                    c=dataPointColor, s=50, alpha=0.5)

        C_transformed = pca.fit_transform(C)
        plt.scatter(C_transformed[:, 0], C_transformed[:, 1], c=centroidColor, s=200, marker=('x'))
        plt.show()

    def __plot_clusters(self,labels):
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