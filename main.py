import json
from cmd import Cmd
from search.tf_idf_search import PlaintextSearch, SklearnSearch
from search.bert_search import BertWordSearch
from search.babbage_search import BabbageSearch
from argparse import ArgumentParser
from data.indicators import Indicators

class SearchShell(Cmd):
    intro = 'Search for matching descriptors given a query'
    prompt = 'Search: '
    file = None

    def __init__(self, *, n=3, text=True, bert=True, babbage=True, sklearn=True):

        super(SearchShell, self).__init__()

        #read descriptions from json array
        #TODO: this should pull from a Corpus for the descriptions, instead of some floating json file
        with open('data/descriptions.json') as f:
            self.descriptions = json.load(f)

        self.text_search = PlaintextSearch(self.descriptions) if text else None
        self.bert_search = BertWordSearch(self.descriptions) if bert else None
        self.babbage_search = BabbageSearch(self.descriptions) if babbage else None
        self.sklearn_search = SklearnSearch(self.descriptions) if sklearn else None
        
        self.n = n


    def do_text(self, arg):
        if self.text_search is None:
            print('Text search not enabled')
            return
        text_results = self.text_search.search(arg, n=self.n)
        self.print_results(text_results, 'text')

    def do_bert(self, arg):
        if self.bert_search is None:
            print('BERT search not enabled')
            return
        bert_results = self.bert_search.search(arg, n=self.n)
        self.print_results(bert_results, 'BERT')

    def do_babbage(self, arg):
        if self.babbage_search is None:
            print('Babbage search not enabled')
            return
        babbage_results = self.babbage_search.search(arg, n=self.n)
        self.print_results(babbage_results, 'babbage')

    def do_sklearn(self, arg):
        if self.sklearn_search is None:
            print('sklearn search not enabled')
            return
        sklearn_results = self.sklearn_search.search(arg, n=self.n)
        self.print_results(sklearn_results, 'sklearn')


    def default(self, arg):
        """Run search on all search engines"""
        if self.text_search is not None:
            text_results = self.text_search.search(arg, n=self.n)
            self.print_results(text_results, 'text')
        
        if self.bert_search is not None:
            bert_results = self.bert_search.search(arg, n=self.n)
            self.print_results(bert_results, 'BERT')
        
        if self.babbage_search is not None:
            babbage_results = self.babbage_search.search(arg, n=self.n)
            self.print_results(babbage_results, 'babbage')

        if self.sklearn_search is not None:
            sklearn_results = self.sklearn_search.search(arg, n=self.n)
            self.print_results(sklearn_results, 'sklearn')


    def print_results(self, results:list[tuple[str,float]], search_type:str):
        print(f'--------------------------------- {search_type} results: ---------------------------------')
        if len(results) == 0:
            print('No results found\n')
            return
        
        for doc, score in results:
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nscore: {score}\n{doc}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n')

  



if __name__ == '__main__':

    #simple arg parser for n=3, text=True, bert=True, babbage=True
    parser = ArgumentParser()
    parser.add_argument('-n', type=int, default=3)
    parser.add_argument('-text', default=False, action='store_true', help='enable text (tf-idf) search')
    parser.add_argument('-bert', default=False, action='store_true', help='enable bert search')
    parser.add_argument('-babbage', default=False, action='store_true', help='enable babbage search')
    parser.add_argument('-sklearn', default=False, action='store_true', help='enable sklearn (tf-idf) search')
    args = parser.parse_args()

    n = args.n
    text = args.text
    bert = args.bert
    babbage = args.babbage
    sklearn = args.sklearn

    #if no engines are specified, default to all
    if not text and not bert and not babbage and not sklearn:
        text, bert, babbage, sklearn = True, True, True, True


    SearchShell(n=n, text=text, bert=bert, babbage=babbage, sklearn=sklearn).cmdloop()