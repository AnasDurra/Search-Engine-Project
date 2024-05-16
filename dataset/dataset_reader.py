from abc import abstractmethod
from collections import defaultdict


class DatasetReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load_as_dict(self) -> dict:
        pass

    @abstractmethod
    def read_queries(self) -> dict:
        pass

    @abstractmethod
    def read_qrels(self) -> defaultdict:
        pass
