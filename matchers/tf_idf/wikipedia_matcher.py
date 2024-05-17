from common.constants import Locations
from matchers.tf_idf.query_matcher import QueryMatcher


class WikipediaMatcher(QueryMatcher):
    def __init__(self):
        super().__init__(Locations.WIKIPEDIA_COLLECTION_NAME)
