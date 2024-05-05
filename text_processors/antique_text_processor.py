from typing import List
from overrides import overrides

from text_processors.base_text_processor import BaseTextProcessor


class AntiqueTextProcessor(BaseTextProcessor):

    @overrides
    def process(self, text) -> List[str]:
        processed_text = self.remove_stopwords(text)
        processed_text = self.lemmatize(processed_text)  # Add lemmatization
        return processed_text
