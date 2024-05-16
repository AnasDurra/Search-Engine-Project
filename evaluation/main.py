import os

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
        return
    #     qrels_path = os.environ.get('ANTIQUE_DATASET_PATH', '../data/antique.txt')
    #     qrels_reader = AntiqueReader(qrels_path)  # Replace AntiqueReader with the appropriate reader for Antique dataset
    #     queries = load_queries_from_database(os.environ.get('ANTIQUE_VECTOR_DATABASE_PATH', 'engines/antique/db'))
    elif dataset_name.lower() == 'wikipedia':
        file_path = "../data/wikipedia/wikipedia-en.csv"
        reader = WikipediaReader(file_path)  # TODO remove filepath from arguments
        qrels = reader.read_qrels()
        queries = reader.read_queries()
    else:
        raise ValueError("Invalid dataset name")

    return qrels, queries


def save_evaluation_results(evaluation_results, dataset_name, do_embedding):
    total_map = sum(metrics_results["AveragePrecisionCalculator"] for metrics_results in evaluation_results.values())
    total_queries = len(evaluation_results)
    average_map = total_map / total_queries

    total_mrr = sum(
        evaluation_results["ReciprocalRankCalculator"] for evaluation_results in evaluation_results.values())
    average_mrr = total_mrr / total_queries

    map_file_path = f"evaluation_results_{dataset_name}_{'with' if do_embedding else 'without'}_embedding_map.csv"
    with open(map_file_path, 'w') as f:
        f.write(f"Average MAP\n{average_map}\n")

    mrr_file_path = f"evaluation_results_{dataset_name}_{'with' if do_embedding else 'without'}_embedding_mrr.csv"
    with open(mrr_file_path, 'w') as f:
        f.write(f"Average MRR\n{average_mrr}\n")

    for metric_name, metric_value in evaluation_results.items():
        if metric_name in ["PrecisionAtKCalculator", "RecallAtKCalculator"]:
            metric_file_path = f"evaluation_results_{dataset_name}_{'with' if do_embedding else 'without'}_embedding_{metric_name.lower()}.csv"
            with open(metric_file_path, 'w') as f:
                f.write("Query ID,Value\n")
                for query_id, value in metric_value.items():
                    f.write(f"{query_id},{value}\n")


def main():
    dataset_name, do_embedding = get_user_input()

    qrels, queries = load_data(dataset_name)

    average_precision_calculator = AveragePrecisionCalculator()
    precision_at_k_calculator = PrecisionCalculator()
    recall_at_k_calculator = RecallCalculator()
    reciprocal_rank_calculator = ReciprocalRankCalculator()

    metric_calculators = [
        average_precision_calculator,
        precision_at_k_calculator,
        recall_at_k_calculator,
        reciprocal_rank_calculator
    ]

    evaluation_manager = EvaluationManager(metric_calculators)

    evaluation_results = evaluation_manager.evaluate(queries, qrels)

    save_evaluation_results(evaluation_results, dataset_name, do_embedding)


if __name__ == "__main__":
    main()
