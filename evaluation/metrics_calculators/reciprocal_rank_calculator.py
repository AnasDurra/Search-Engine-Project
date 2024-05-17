from typing import List, Dict, Optional

from evaluation.metrics_calculators.metric_calculator import MetricCalculator


class ReciprocalRankCalculator(MetricCalculator):
    def calculate(self, query_id: str, retrieved_docs: List[str], qrels: Dict[str, Dict[str, int]],
                  k: Optional[int] = None) -> float:

        relevant_docs = qrels.get(query_id, {})

        if k is not None:
            retrieved_docs = retrieved_docs[:k]

        for i, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc['doc_id']
            if doc_id in relevant_docs.keys() and relevant_docs[doc_id] > 0:
                return 1.0 / i
        return 0.0
