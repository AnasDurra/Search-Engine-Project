from overrides import overrides

from dataset.dataset_reader import DatasetReader


class WikipidiaReader(DatasetReader):

    @overrides
    def load_as_dict(self) -> dict:
        # TODO: WRITE YOUR OWN LOGIC
        pass
