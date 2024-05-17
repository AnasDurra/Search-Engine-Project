from typing import List
from overrides import overrides

from text_processors.base_text_processor import BaseTextProcessor


class AntiqueTextProcessor(BaseTextProcessor):

    @overrides
    def process(self, text) -> List[str]:
        text = text.lower()
        tokens = self._word_tokenizer(text)
        tokens = self._remove_punctuations(tokens)
        tokens = self.remove_apostrophe(tokens)
        tokens = self._remove_stopwords(tokens)
        tokens = self.remove_markers(tokens)
        tokens = self.stemming(tokens)
        tokens = self.replace_under_score_with_space(tokens)
        # tokens = self._spell_check(tokens)
        tokens = self._remove_whitespaces(tokens)
        processed_text = self._lemmatize(tokens)
        return processed_text
