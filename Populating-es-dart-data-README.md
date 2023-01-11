
# Populating ElasticSearch Scripts

### Dart Documents

Download the data from Google Drive:

https://drive.google.com/drive/folders/1bUNoOTfefmYRuEWuaZhfOqQIyyMV72QE?usp=sharing

Follow the instructions on where to put each file, using this slack message:
https://jataware.slack.com/archives/CAZTFECQY/p1671563821874239

### Additional Dependencies

Requirements.txt contains dependencies to handle and create embeddings, but you'll need to
also have the `elasticsearch` python library available in your environment.

### Prerequisites

Create the `documents` and 	`document_paragraphs` indexes in elasticsearch:

```
PUT /documents
{
  "mappings": {
    "properties": {
      }
    }
  }
}
```

```
PUT /document_paragraphs
{
  "mappings": {
    "properties": {
      "embeddings": {
        "type": "dense_vector",
        "dims": 768
      }
    }
  }
}
```

## Scripts

The script files point to hardcoded local elasticsearch urls.

To upload document metadata to es, run:

`python upload_all_document_metadata.py`

To upload paragraph text and embeddings to es:

`python upload_paragraphs_embeddings.py`
