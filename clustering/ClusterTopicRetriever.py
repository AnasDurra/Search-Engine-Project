import os

from dotenv import load_dotenv

load_dotenv()


class ClusterTopicRetriever:
    def __init__(self, model_name):
        self.model_name = model_name
        cluster_dir_path = os.environ.get('TOPICS_PATH')
        self.cluster_dir = f"{cluster_dir_path}/{model_name}_clusters"

    def get_topic_file(self, cluster_number):
        cluster_dir = f"{self.cluster_dir}"
        print(cluster_dir)
        topic_file = os.path.join(cluster_dir, f'cluster{cluster_number}_topics.txt')
        if not os.path.isfile(topic_file):
            raise FileNotFoundError(f'Topic file for cluster {cluster_number} not found.')
        with open(topic_file, 'r') as f:
            topics = f.readlines()
        return [line.strip() for line in topics]


if __name__ == "__main__":
    model_name = "antique"
    cluster_number = 1

    topic_retriever = ClusterTopicRetriever(model_name)

    topics = topic_retriever.get_topic_file(cluster_number)
    print(f"Topics for cluster {cluster_number}:\n", "\n".join(topics))
