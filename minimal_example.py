from data.indicators import Indicators
from search.bert_search import BertWordSearch, BertSentenceSearch
# from search.tf_idf_search import PlaintextSearch # alternate search engine


# load a corpus of documents, and set up a search engine
corpus = Indicators.get_corpus()
engine = BertWordSearch(corpus) # pass cuda=False to run on CPU instead of (default) GPU

# concept to search for in the corpus
query = 'number of people who have been vaccinated'
print('Query:', query)

# example of generating the query matrix (embedding) manually. search does this internally
embedding = engine.embed_query(query)
print('embedding:\n', embedding, '\n')

# perform the search, and collect the top 10 matches
matches = engine.search(query, n=10)

# print the results of the search
print('Top 10 matches:')
for match_id, score in matches:
    raw_text = corpus[match_id]   # get the matching text for the given id
    print(raw_text, end='\n\n\n') # print the text







