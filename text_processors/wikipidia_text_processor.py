from typing import List

from overrides import overrides

from text_processors.base_text_processor import BaseTextProcessor


class WikipidiaTextProcessor(BaseTextProcessor):

    @overrides
    def process(self, text) -> List[str]:
        # TODO: ADD YOUR OWN IMPLEMENTATION
        pass

    # TODO: DEFINE OTHER TEXT PROCESSING METHODS
