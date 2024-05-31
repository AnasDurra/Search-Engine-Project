import os
from dotenv import load_dotenv
from dataset.WikipediaReader import WikipediaReader
from dataset.antique_reader import AntiqueReader
from evaluation.evaluation_manager import EvaluationManager
from evaluation.metrics_calculators.average_precision_calculator import AveragePrecisionCalculator
from evaluation.metrics_calculators.precision_calculator import PrecisionCalculator
from evaluation.metrics_calculators.recall_calculator import RecallCalculator
from evaluation.metrics_calculators.reciprocal_rank_calculator import ReciprocalRankCalculator
from matchers.embedding.antique_embedding_matcher import AntiqueEmbeddingMatcher
from matchers.embedding.wikipedia_embedding_matcher import WikipediaEmbeddingMatcher
from matchers.tf_idf.antique_matcher import AntiqueMatcher
from matchers.tf_idf.wikipedia_matcher import WikipediaMatcher


def get_user_input():
    dataset_name = input("Enter dataset name (e.g., 'antique' or 'wikipedia'): ").strip().lower()
    do_embedding = input("Do you want to perform word embedding? (yes/no): ").strip().lower() == "yes"
    return dataset_name, do_embedding


def load_data(dataset_name):  # TODO: embedding
    if dataset_name.lower() == 'antique':
        file_path = os.environ.get('ANTIQUE_DATASET_PATH')
        reader = AntiqueReader(file_path)
        qrels = reader.read_qrels()
        queries = reader.read_queries()
    elif dataset_name.lower() == 'wikipedia':
        file_path = os.environ.get('WIKIPEDIA_DATASET_PATH')
        reader = WikipediaReader(file_path)
        qrels = reader.read_qrels()
        queries = reader.read_queries()
    else:
        raise ValueError("Invalid dataset name")
    return qrels, queries


def calculate_average_metrics(evaluation_results):
    total_map = sum(metrics_results["AveragePrecisionCalculator"] for metrics_results in evaluation_results.values())
    total_mrr = sum(metrics_results["ReciprocalRankCalculator"] for metrics_results in evaluation_results.values())
    total_queries = len(evaluation_results)

    average_map = total_map / total_queries if total_queries > 0 else 0.0
    average_mrr = total_mrr / total_queries if total_queries > 0 else 0.0

    return average_map, average_mrr


def save_evaluation_results(evaluation_results, dataset_name, do_embedding, average_map, average_mrr):
    embedding_folder = 'with_embedding' if do_embedding else 'without_embedding'
    directory = os.path.join('results', dataset_name, embedding_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    map_file_path = os.path.join(directory, "average_map.csv")
    with open(map_file_path, 'w') as f:
        f.write(f"Average MAP\n{average_map:.6f}\n")

    mrr_file_path = os.path.join(directory, "average_mrr.csv")
    with open(mrr_file_path, 'w') as f:
        f.write(f"Average MRR\n{average_mrr:.6f}\n")

    max_query_id_length = max(len(query_id) for query_id in evaluation_results.keys())

    for metric_name in ["PrecisionCalculator", "RecallCalculator"]:
        metric_file_path = os.path.join(directory, f"{metric_name.lower()}_at_k.csv")
        if os.path.exists(metric_file_path):
            os.remove(metric_file_path)
        with open(metric_file_path, 'w') as f:
            f.write("Query ID".ljust(max_query_id_length + 5) + "Value\n")
            for query_id, metrics in evaluation_results.items():
                metric_value = metrics[metric_name]
                # print(metric_value)
                f.write(f"{query_id.ljust(max_query_id_length + 5)}, {metric_value:.6f}\n")


def main():
    dataset_name, do_embedding = get_user_input()
    qrels, queries = load_data(dataset_name)

    average_precision_calculator = AveragePrecisionCalculator()
    precision_calculator = PrecisionCalculator()
    recall_calculator = RecallCalculator()
    reciprocal_rank_calculator = ReciprocalRankCalculator()

    metric_calculators = [
        average_precision_calculator,
        precision_calculator,
        recall_calculator,
        reciprocal_rank_calculator
    ]

    if dataset_name == 'antique':
        matcher = AntiqueEmbeddingMatcher() if do_embedding else AntiqueMatcher()
    elif dataset_name == 'wikipedia':
        matcher = WikipediaEmbeddingMatcher() if do_embedding else WikipediaMatcher()
    else:
        raise ValueError("Invalid dataset name")

    evaluation_manager = EvaluationManager(metric_calculators, matcher)

    k = int(os.environ.get('RECALL_PRECISION_THRESHOLD', 10))
    evaluation_results = evaluation_manager.evaluate(queries, qrels, k)

    average_map, average_mrr = calculate_average_metrics(evaluation_results)
    save_evaluation_results(evaluation_results, dataset_name, do_embedding, average_map, average_mrr)


if __name__ == "__main__":
    load_dotenv()
    main()
