from overrides import overrides
from dataset.dataset_reader import DatasetReader


class AntiqueReader(DatasetReader):
    @overrides
    def load_as_dict(self) -> dict:
        key_value_pairs = {}
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                key, value = line.strip().split('\t')
                key_value_pairs[key] = value
        return key_value_pairs
