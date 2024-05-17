from common.constants import Locations
from matchers.tf_idf.query_matcher import QueryMatcher


class AntiqueMatcher(QueryMatcher):
    def __init__(self):
        super().__init__(Locations.ANTIQUE_COLLECTION_NAME)
