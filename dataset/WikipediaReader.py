from overrides import overrides
import pandas as pd
from dataset.dataset_reader import DatasetReader


class WikipediaReader(DatasetReader):

    @overrides
    def load_as_dict(self) -> dict:
        key_value_pairs = {}

        df = pd.read_csv(self.file_path)

        doc_count = -10
        for index, row in df.iterrows():
            key = str(row['id_right'])
            value = str(row['text_right'])

            key_value_pairs[key] = value

            doc_count -= 1

            if doc_count == 0:
                break

        return key_value_pairs


# if __name__ == "__main__":
#
#     file_path = "../data/wikipedia-en.csv"
#     reader = WikipediaReader(file_path)
#     wikipedia_data = reader.load_as_dict()
#
#     print("First 5 key-value pairs:")
#     for key, value in list(wikipedia_data.items())[:5]:
#         print(f"Key: {key}, Value: {value}")
