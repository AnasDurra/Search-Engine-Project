from typing import List, Dict, Optional

from evaluation.metrics_calculators.metric_calculator import MetricCalculator


class PrecisionCalculator(MetricCalculator):
    def calculate(self, query_id: str, retrieved_docs: List[str], qrels: Dict[str, Dict[str, int]],
                  k: Optional[int] = None) -> float:
        if not retrieved_docs:
            return 0.0

        relevant_docs = qrels.get(query_id, {})
        relevant_retrieved = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_docs)

        if not relevant_retrieved:
            return 0.0
        # print('query: ',query_id)
        # print('precision: ',relevant_retrieved / min(len(retrieved_docs), k))
        return relevant_retrieved / min(len(retrieved_docs), k)
