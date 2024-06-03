from typing import List
from overrides import overrides

from text_processors.base_text_processor import BaseTextProcessor


class AntiqueTextProcessor(BaseTextProcessor):

    @overrides
    def process(self, text) -> List[str]:
        tokens = self._word_tokenizer(text)
        tokens = self._lowercase_tokens(tokens)
        tokens = self._strip_punctuation(tokens)
        tokens = self._remove_apostrophes(tokens)
        tokens = self._filter_stop_words(tokens)
        tokens = self._remove_registered_markers(tokens)
        tokens = self._lemmatize_tokens(tokens)
        tokens = self._process_abbreviations(tokens)
        tokens = self._lowercase_tokens(tokens)
        tokens = self._eliminate_whitespaces(tokens)
        return tokens
