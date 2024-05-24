from matchers.embedding.base_embedding_matcher import BaseEmbeddingMatcher
from text_processors.antique_text_processor import AntiqueTextProcessor
from text_processors.wikipedia_text_processor import WikipediaTextProcessor


class WikipediaEmbeddingMatcher(BaseEmbeddingMatcher):

    def __init__(self):
        super().__init__(
            model_name='wikipedia',
            text_processor=WikipediaTextProcessor(),
        )
