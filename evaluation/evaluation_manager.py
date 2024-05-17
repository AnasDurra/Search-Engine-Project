from typing import Dict, List, Optional
from evaluation.metrics_calculators.metric_calculator import MetricCalculator


class EvaluationManager:
    def __init__(self, metric_calculators: List[MetricCalculator], matcher):
        self.metric_calculators = metric_calculators
        self.matcher = matcher

    def evaluate(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], k: Optional[int] = None) -> Dict[
        str, Dict[str, float]]:
        evaluation_results = {}

        for query_id, query_text in queries.items():
            retrieved_docs = self.matcher.match(query_text)
            metrics_results = {}
            for metric_calculator in self.metric_calculators:
                metric_name = metric_calculator.__class__.__name__
                if metric_name in ["AveragePrecisionCalculator", "RecallCalculator", "PrecisionCalculator"]:
                    retrieved_doc_ids = [doc_info['doc_id'] for doc_info in retrieved_docs]
                    metric_value = metric_calculator.calculate(query_id, retrieved_doc_ids, qrels, k=k)
                else:
                    metric_value = metric_calculator.calculate(query_id, retrieved_docs, qrels)
                metrics_results[metric_name] = metric_value

            evaluation_results[query_id] = metrics_results

        return evaluation_results  # {query_id: {metric_name:value}}
