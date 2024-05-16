from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class MetricCalculator(ABC):
    @abstractmethod
    def calculate(self, query_id: str, retrieved_docs: List[str], qrels: Dict[str, Dict[str, int]], k: Optional[int] = None) -> float:
        pass
