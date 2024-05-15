from matchers.embedding.base_embedding_matcher import BaseEmbeddingMatcher
from text_processors.antique_text_processor import AntiqueTextProcessor


class AntiqueEmbeddingMatcher(BaseEmbeddingMatcher):

    def __init__(self):
        super().__init__(
            model_name='antique',
            text_processor=AntiqueTextProcessor()
        )
