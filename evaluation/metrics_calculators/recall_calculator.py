from typing import List, Dict, Optional

from evaluation.metrics_calculators.metric_calculator import MetricCalculator


class RecallCalculator(MetricCalculator):
    def calculate(self, query_id: str, retrieved_docs: List[str], qrels: Dict[str, Dict[str, int]], k: Optional[int] = None) -> float:
        relevant_docs = qrels.get(query_id, {})
        relevant_retrieved = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_docs)
        total_relevant = sum(relevant_docs.values())
        return relevant_retrieved / total_relevant if total_relevant > 0 else 0
