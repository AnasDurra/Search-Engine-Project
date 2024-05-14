from abc import abstractmethod


class DatasetReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load_as_dict(self) -> dict:
        pass
