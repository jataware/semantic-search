import json
from neural_tf_idf import NeuralSearch, PlaintextSearch
from babbage_search import BabbageSearch

def main():

    #read descriptions from json array
    with open('descriptions.json') as f:
        descriptions = json.load(f)

    text_search = PlaintextSearch(descriptions)
    # neural_search = NeuralSearch(descriptions)
    babbage_search = BabbageSearch(descriptions)
    
    while True:
        query = input('Search: ')
        text_results = text_search.search(query, n=3)
        print_results(text_results, 'text')
        
        # neural_results = neural_search.search(query, n=3)
        # print_results(neural_results, 'neural')
        
        babbage_results = babbage_search.search(query, n=3)
        print_results(babbage_results, 'babbage')
    



def print_results(results:list[tuple[str,float]], search_type:str):
    print(f'--------------------------------- {search_type} results: ---------------------------------')
    if len(results) == 0:
        print('No results found\n')
        return
    
    for doc, score in results:
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nscore: {score}\ndoc: {doc}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n\n')





if __name__ == '__main__':
    main()