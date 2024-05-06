from typing import List
from overrides import overrides

from text_processors.base_text_processor import BaseTextProcessor


class AntiqueTextProcessor(BaseTextProcessor):

    @overrides
    def process(self, text) -> List[str]:
        tokens = self.word_tokenizer(text)
        tokens = self.remove_stopwords(tokens)
        processed_text = self.lemmatize(tokens)
        return processed_text
