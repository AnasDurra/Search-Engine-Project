from typing import List
from overrides import overrides
from text_processors.base_text_processor import BaseTextProcessor

class AntiqueTextProcessor(BaseTextProcessor):

    @overrides
    def process(self, text) -> List[str]:
        processed_text = super().process(text)

        return processed_text