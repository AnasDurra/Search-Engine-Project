from fastapi import FastAPI
from matchers.tf_idf.wikipedia_matcher import WikipediaMatcher
from matchers.tf_idf.antique_matcher import AntiqueMatcher
from matchers.embedding.wikipedia_embedding_matcher import WikipediaEmbeddingMatcher
from matchers.embedding.antique_embedding_matcher import AntiqueEmbeddingMatcher
from dotenv import load_dotenv


from .dtos import Dataset
from .dtos import Model
from .dtos import QueryDto

from bson import ObjectId

load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"key": "value"}


def serialize(item):
    if isinstance(item, ObjectId):
        return str(item)
    elif isinstance(item, list):
        return [serialize(sub_item) for sub_item in item]
    elif isinstance(item, dict):
        return {key: serialize(value) for key, value in item.items()}
    return item


@app.post("/query")
async def query(query_dto: QueryDto):
    if query_dto.dataset == Dataset.antique:
        if query_dto.model == Model.tfidf:
            q = AntiqueMatcher()
            output = q.match(query_dto.query)
            serializable_output = serialize(output)
            return serializable_output
        if query_dto.model == Model.embedding:
            q = AntiqueEmbeddingMatcher()
            output = q.match(query_dto.query)
            serializable_output = serialize(output)
            return serializable_output
    if query_dto.dataset == Dataset.wiki:
        if query_dto.model == Model.tfidf:
            q = WikipediaMatcher()
            output = q.match(query_dto.query)
            serializable_output = serialize(output)
            return serializable_output
        if query_dto.model == Model.embedding:
            q = WikipediaEmbeddingMatcher()
            output = q.match(query_dto.query)
            serializable_output = serialize(output)
            return serializable_output