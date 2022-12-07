from search.corpora import Indicators
from search.bert_search import BertSentenceSearch
from elasticsearch import Elasticsearch
import json

es = Elasticsearch("http://localhost:9200")

print(es.info())

# load a corpus of documents, and set up a search engine
corpus = Indicators.get_corpus()
engine = BertSentenceSearch(corpus, cuda=False)

# hardcoded query to search through es feature documents
query = 'number of people who have been vaccinated'
print('Query:', query)

# Embedding for input query
embedding = engine.embed_query(query)

es_query = {
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                "params": {
                    "query_vector": embedding.tolist()
                }
            }
        }
    },
    "_source": [
        "unit",
        "is_primary",
        "name",
        "description",
        "unit_description",
        "display_name",
        "type"
    ]
}

results = json.dumps(es.search(index="features", body=es_query)["hits"]["hits"])

print(f"Results:\n{results}")
