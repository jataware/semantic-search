from search.corpora import Indicators
from search.bert_search import BertSearch
# from search.tf_idf_search import PlaintextSearch # alternate search engine


# load a corpus of documents, and set up a search engine
corpus = Indicators.get_corpus()
engine = BertSearch(corpus)

# perform the search
query = 'number of people who have been vaccinated'
matches = engine.search(query, n=10)

# print the results
print('Query:', query)
print('Top 10 matches:')
for match_id, score in matches:
    raw_text = corpus[match_id]   #get the matching text for the given id
    print(raw_text, end='\n\n\n') #print the text