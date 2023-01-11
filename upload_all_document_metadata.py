from data.dart_papers import DartPapers

from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

print(es.info())

document_metadata = DartPapers.get_metadata()

def index_all_documents():
    """
    Loops through Dart Document metadata in the corpus and uploads to elasticsearch `documents`.
    """
    parsedCount = 1

    for id, metadata in document_metadata.items():
        print(f"{parsedCount} Processing document id: {id}")
        es.index(index="documents", body=metadata, id=id)
        parsedCount += 1


if __name__ == "__main__":
    index_all_documents()
    exit(0)

