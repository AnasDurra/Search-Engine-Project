import os
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from common.constants import Locations
from common.file_utilities import FileUtilities

load_dotenv()


class TopicModeler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.cluster_dir = f"D:/Users/anas/Desktop/IR Project/search-engine/clustering/clusters/{model_name}_clusters"

        model_path = Locations.generate_model_path(model_name)
        self.tfidf_vectorizer = FileUtilities.load_file(model_path)

    def load_cluster(self, cluster_file):
        return FileUtilities.load_file(cluster_file)

    def fit_lda_model(self, doc_term_matrix, num_topics=5):
        lda_topic_model = LatentDirichletAllocation(n_components=num_topics, random_state=12345)
        doc_topic_matrix = lda_topic_model.fit_transform(doc_term_matrix)
        return lda_topic_model, doc_topic_matrix

    def display_top_words(self, lda_model, num_words=10):
        top_words = []
        for topic, words in enumerate(lda_model.components_):
            sorted_words = words.argsort()[::-1]
            topic_words = []
            for i in range(num_words):
                word = self.tfidf_vectorizer.get_feature_names_out()[sorted_words[i]]
                word_weight = words[sorted_words[i]]
                topic_words.append(f'{word} ({word_weight:.3f})')
            top_words.append(f'Topic {topic + 1:02d}: ' + ', '.join(topic_words))
        return top_words

    def save_topics(self, top_words, output_file):
        with open(output_file, 'w') as f:
            for line in top_words:
                f.write(line + '\n')

    def process_clusters(self, num_topics=5, num_words=10):
        cluster_files = [f for f in os.listdir(self.cluster_dir) if f.endswith('.pkl')]
        for cluster_file in cluster_files:
            cluster_path = os.path.join(self.cluster_dir, cluster_file)
            (sparse_matrix, cluster_indices) = self.load_cluster(cluster_path)
            print(sparse_matrix)
            lda_model, doc_topic_matrix = self.fit_lda_model(sparse_matrix, num_topics=num_topics)
            top_words = self.display_top_words(lda_model, num_words=num_words)

            output_file = cluster_path.replace('.pkl', '_topics.txt')
            self.save_topics(top_words, output_file)
            print(f"Processed and saved topics for {cluster_file}")


if __name__ == "__main__":
    model_name = "antique"
    topic_modeler = TopicModeler(model_name)
    topic_modeler.process_clusters(num_topics=5, num_words=10)