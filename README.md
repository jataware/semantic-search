# Neural Semantic Search via LLM Embeddings

This library implements semantic search given a query over a corpus of text documents. The goal is to provide a high quality search engine by leveraging advances in Large Language Models (LLMs). The library provides an interface for representing corpora of text documents, as well as an interface for search engines that search over a corpus. 

## Requirements
See [`requirements.txt`](requirements.txt) for a list of required packages. The library requires **python 3.9** or higher.


## [Minimal Example](minimal_example.py)
```python
from data.indicators import Indicators
from search.bert_search import BertSentenceSearch

# load a corpus of documents, and set up a search engine
corpus = Indicators.get_corpus()
engine = BertSentenceSearch(corpus, save_name=Indicators.__name__)

# concept to search for in the corpus
query = 'number of people who have been vaccinated'
print('Query:', query)

# perform the search, and collect the top 10 matches
matches = engine.search(query, n=10)

# print the results of the search
print('Top 10 matches:')
for match_id, score in matches:
    raw_text = corpus[match_id]   # get the matching text for the given id
    print(raw_text, end='\n\n\n') # print the text
```

## Corpus
A corpus represents a collection of documents as well as IDs that map to the documents. Corpora implement a concrete version of the [Corpus](data/corpora.py) abstract class. A corpus can be thought of as a `dict[T, str]` where `T` is some sort of document ID. Several corpora are provided in the [data](data/) module via implementations of the `CorpusLoader` abstract class, which provide a `get_corpus()` method that returns a `Corpus` instance.

### [Indicators](data/indicators.py)
a collection of descriptions of features largely drawn from the World Development Indicators (WDI) dataset. Each document id is a tuple of the `(dataset_id, feature_name)`, specifying which specific dataset, and which feature column the description is for. The descriptions are formatted as follows:
```
name: <feature_name>;
display name: <feature_name>;
description: <description>;
unit: <unit>;
unit description: <unit_description>;
```

for example
```
name: anomaly-hd40-annual-mean;
display name: Predicted Annual Mean Number of Very Hot Days Anomaly (days);
description: Annual Predicted Mean Number of Very Hot Days Anomaly (Tmax>40degreesC) - Predicted Data (CMIP - using 1986-2005 as reference period, RCP8.5 High Emissions scenario, and Ensemble model) - mean over the period 2020-2039 (showing the date of 2020 for the predicted mean number of very hot days anomaly during that period).;
unit: days;
unit description: days;
```

### [DART Papers](data/dart_papers.py)
A collection of research papers used in the DART project. Since papers are generally longer than LLMs can accommodate, they are split up into chunks (currently by paragraph, but perhaps this will be changed to by sentence). Each document id is a tuple of the `(paper_id, chunk_index)`, specifying which specific paper, and which chunk of the paper the description is for. Descriptions are the raw text of the chunk, and are not formatted in any way.

NOTE: to use this corpus, you must separately download the DART papers (~1.3GB), and save them to `data/dart_cdr.json_mar_2022`.

### [World Modelers Ontology Nodes](data/wm_ontology.py)
Mainly for convenience, this corpus provides all of the nodes in the World Modelers flat ontology and additional terms that are examples of each node. This provides a convenient way to use nodes as a search query. Each document id is simple the node name, and the description is a comma separated list of the node name and all of its examples. Some examples:
```
"labor" -> "labor, casual labour, decent work, good job, hard work, hired labour, job activities, labor, labour, labour force, labour market, wage, work, workload, workplace"

"small_businesses" -> "small businesses, entrepreneurs, entrepreneurship, small businesses"

"election" -> "election, ballot, caucus, election, referendum, voting"
```


## Corpus Example Usage
```python
from data.dart_papers import DartPapers
# from data.wm_ontology import FlatOntology

corpus = Indicators.get_corpus()

for doc_id, doc in corpus.items():
    print(doc_id, doc)
```

## Search Engine
So far the following search engines are implemented:
- [tf-idf search](search/tf_idf_search.py) 
  - **SklearnSearch**: tf-idf based on sklearn feature extraction
  - **PlaintextSearch**: manual numpy implementation of tf-idf, mainly a reference for the neural-tf-idf implementation
  - ref: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- [BertSearch](search/bert_search.py)
  - **BertWordSearch**: computes word level embeddings with an LLM, and then uses a soft version of tf-idf to compute document rankings for a given query
  - **BertSentenceSearch** version: computes sentence level embeddings with an LLM, and then uses cosine similarity to compute document rankings for a given query
- [BabbageSearch](search/babbage_search.py)
  - **BabbageSearch**: similar to BertSentenceSearch, but uses OpenAIs Babbage model, which leverages GPT-3, instead of a BERT model. Requires an OpenAI API key to be set in the environment variable `OPENAI_API_KEY`

## Experiments
This library includes several experiments to compare the performance of LLM search engines to results from UAZ.

### [Ontology to Indicator Mapping](experiments/uaz_indicators_comparison.py)
For each concept node in the world modelers ontology, finds the top 3 matching indicators from the WDI datasets that match that concept. The results are then compared to the top 3 results from UAZ (noting that UAZ provides an inverted view of their results, mapping from indicator to top concepts)

### [Ontology to Paper Mapping](experiments/uaz_documents_comparison.py)
(currently WIP) For each concept node in the world modelers ontology, finds the top 3 matching papers from the DART corpus that match that concept. The results are then compared to the results from UAZ concept graph, which links ontology nodes to other nodes, based on evidence from the papers. Eventually, this library will add approaches for pairing concept nodes, computing their causal directionality, and computing their polarity.