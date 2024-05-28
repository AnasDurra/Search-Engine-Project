from enum import Enum
from pydantic import BaseModel


class Model(str, Enum):
    tfidf = "tfidf"
    embedding: "embedding"


class Dataset(str, Enum):
    wiki = "wiki"
    antique = "antique"


class QueryDto(BaseModel):
    dataset: Dataset
    model: Model
    query: str
