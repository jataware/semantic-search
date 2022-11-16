import json
from tf_idf_search import PlaintextSearch
from bert_search import BertSearch
from babbage_search import BabbageSearch

def main():

    #read descriptions from json array
    with open('data/descriptions.json') as f:
        descriptions = json.load(f)

    # initialize search objects to None (i.e. don't use them)
    text_search, bert_search, babbage_search = None, None, None

    # COMMENT out search here to skip it during loop
    text_search = PlaintextSearch(descriptions)
    bert_search = BertSearch(descriptions)
    babbage_search = BabbageSearch(descriptions)
    
    while True:
        query = input('Search: ')
        
        if text_search is not None:
            text_results = text_search.search(query, n=3)
            print_results(text_results, 'text')
        
        if bert_search is not None:
            bert_results = bert_search.search(query, n=3)
            print_results(bert_results, 'BERT')
        
        if babbage_search is not None:
            babbage_results = babbage_search.search(query, n=3)
            print_results(babbage_results, 'babbage')
    



def print_results(results:list[tuple[str,float]], search_type:str):
    print(f'--------------------------------- {search_type} results: ---------------------------------')
    if len(results) == 0:
        print('No results found\n')
        return
    
    for doc, score in results:
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nscore: {score}\ndoc: {doc}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n')





if __name__ == '__main__':
    main()