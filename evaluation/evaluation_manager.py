from typing import List, Dict

from evaluation.metrics_calculators.metric_calculator import MetricCalculator


class EvaluationManager:
    def __init__(self, metric_calculators: List[MetricCalculator]):
        self.metric_calculators = metric_calculators

    def evaluate(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], retrieved_docs_list: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        evaluation_results = {}

        for query_id, query_text in queries.items():
            retrieved_docs = [doc['doc_id'] for doc in retrieved_docs_list]
            metrics_results = {}
            for metric_calculator in self.metric_calculators:
                metric_name = metric_calculator.__class__.__name__
                metric_value = metric_calculator.calculate(query_id, retrieved_docs, qrels)
                metrics_results[metric_name] = metric_value

            evaluation_results[query_id] = metrics_results

        return evaluation_results
