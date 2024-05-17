import os

from dotenv import load_dotenv

from dataset.WikipediaReader import WikipediaReader
from dataset.antique_reader import AntiqueReader
from evaluation.evaluation_manager import EvaluationManager
from evaluation.metrics_calculators.average_precision_calculator import AveragePrecisionCalculator
from evaluation.metrics_calculators.precision_calculator import PrecisionCalculator
from evaluation.metrics_calculators.recall_calculator import RecallCalculator
from evaluation.metrics_calculators.reciprocal_rank_calculator import ReciprocalRankCalculator


def get_user_input():
    dataset_name = input("Enter dataset name (e.g., 'antique' or 'wikipedia'): ").strip().lower()
    do_embedding = input("Do you want to perform word embedding? (yes/no): ").strip().lower() == "yes"
    return dataset_name, do_embedding


def load_data(dataset_name):
    if dataset_name.lower() == 'antique':
        file_path = "../data/antique/antique.txt"
        reader = AntiqueReader(file_path)  # TODO remove filepath from arguments
        qrels = reader.read_qrels()
        queries = reader.read_queries()
    elif dataset_name.lower() == 'wikipedia':
        file_path = "../data/wikipedia/wikipedia-en.csv"
        reader = WikipediaReader(file_path)  # TODO remove filepath from arguments
        qrels = reader.read_qrels()
        queries = reader.read_queries()
    else:
        raise ValueError("Invalid dataset name")

    return qrels, queries


def save_evaluation_results(evaluation_results, dataset_name, do_embedding):
    directory = f"results_{dataset_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    total_map = sum(metrics_results["AveragePrecisionCalculator"] for metrics_results in evaluation_results.values())
    total_queries = len(evaluation_results)
    average_map = total_map / total_queries

    total_mrr = sum(
        evaluation_results["ReciprocalRankCalculator"] for evaluation_results in evaluation_results.values())
    average_mrr = total_mrr / total_queries

    map_file_path = f"{directory}/{'with' if do_embedding else 'without'}_embedding_map.csv"
    with open(map_file_path, 'w') as f:
        f.write(f"Average MAP\n{average_map}\n")

    mrr_file_path = f"{directory}/{'with' if do_embedding else 'without'}_embedding_mrr.csv"
    with open(mrr_file_path, 'w') as f:
        f.write(f"Average MRR\n{average_mrr}\n")

    print(evaluation_results)
    max_query_id_length = max(len(query_id) for query_id in evaluation_results.keys())

    for metric_name in ["PrecisionCalculator", "RecallCalculator"]:
        metric_file_path = f"{directory}/{'with' if do_embedding else 'without'}_embedding_{metric_name.lower()}.csv"
        if os.path.exists(metric_file_path):
            os.remove(metric_file_path)

    for query_id, metrics in evaluation_results.items():
        for metric_name, metric_value in metrics.items():
            if metric_name in ["PrecisionCalculator", "RecallCalculator"]:
                metric_file_path = f"{directory}/{'with' if do_embedding else 'without'}_embedding_{metric_name.lower()}.csv"
                mode = 'a' if os.path.exists(metric_file_path) else 'w'
                with open(metric_file_path, mode) as f:
                    if mode == 'w':
                        f.write("Query ID".ljust(
                            max_query_id_length + 5) + "Value\n")  # Fixed width for query ID column plus additional space
                    f.write(
                        f"{query_id.ljust(max_query_id_length + 5)}, {metric_value:.6f}\n")  # Fixed width for query ID column plus additional space, 6 decimal places for value
                print(f"Appended {metric_name} result for query {query_id} to {metric_file_path}")


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

    # Specify the matcher type based on the dataset
    matcher_type = 'antique' if dataset_name.lower() == 'antique' else 'wikipedia'

    evaluation_manager = EvaluationManager(metric_calculators, matcher_type)

    evaluation_results = evaluation_manager.evaluate(queries, qrels, 9)

    save_evaluation_results(evaluation_results, dataset_name, do_embedding)


if __name__ == "__main__":
    load_dotenv()
    main()
