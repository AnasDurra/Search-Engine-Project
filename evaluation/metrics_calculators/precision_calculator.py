from typing import List, Dict, Optional

from evaluation.metrics_calculators.metric_calculator import MetricCalculator


class PrecisionCalculator(MetricCalculator):
    def calculate(self, query_id: str, retrieved_docs: List[str], qrels: Dict[str, Dict[str, int]], k: Optional[int] = None) -> float:
        relevant_docs = qrels.get(query_id, {})
        relevant_retrieved = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_docs)
        return relevant_retrieved / min(len(retrieved_docs), k)
