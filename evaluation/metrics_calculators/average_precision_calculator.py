from typing import List, Dict, Optional

from evaluation.metrics_calculators.metric_calculator import MetricCalculator


class AveragePrecisionCalculator(MetricCalculator):
    def calculate(self, query_id: str, retrieved_docs: List[str], qrels: Dict[str, Dict[str, int]],
                  k: Optional[int] = None) -> float:
        if query_id not in qrels:
            return 0.0

        relevant_docs = qrels[query_id]
        print('relevant: ', relevant_docs)
        num_relevant_docs = 0

        # if num_relevant_docs == 0:
        #     return 0.0

        num_retrieved_relevant_docs = 0
        sum_precisions = 0.0

        print(len(retrieved_docs))
        for i, doc_info in enumerate(retrieved_docs, start=1):
            doc_id = doc_info['doc_id']
            print(doc_id)
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                num_retrieved_relevant_docs += 1
                num_relevant_docs += 1
                print(num_relevant_docs)
                precision_at_i = num_retrieved_relevant_docs / i
                sum_precisions += precision_at_i

        average_precision = 0 if num_relevant_docs == 0 else sum_precisions / num_relevant_docs
        print(average_precision)
        return average_precision
