import json
from transformers import BertTokenizer, BertModel
from abc import ABC, abstractmethod
import re

import pdb

def main():

    with open('indicators.jsonl') as f:
        lines = f.readlines()
        indicators = [json.loads(line) for line in lines]
    
    descriptions = [indicator['_source']['description'] for indicator in indicators]
    
    
    text_search = PlaintextSearch(descriptions)
    
    while True:
        query = input('Search: ')
        text_results = text_search.search(query)
        print_results(text_results)
    
    
    
    results = text_search.search('precipitation')


    # neural_search = NeuralSearch(descriptions)
    
    
    pdb.set_trace()


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    pdb.set_trace()




def print_results(results:list[tuple[str,float]]):
    if len(results) == 0:
        print('No results found\n')
        return
    
    for doc, score in results:
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nscore: {score}\ndoc: {doc}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n\n')


class TF_IDF(ABC):
    @abstractmethod
    def __init__(self, raw_corpus:list[str]): ...

    @abstractmethod
    def search(self, query:str) -> list[tuple[str, float]]: ...


class PlaintextSearch(TF_IDF):
    """simple text based implementation of TF-IDF"""
        
    def __init__(self, corpus:list[str]):
        self.corpus = corpus
        self._build_tf_idf()

    def _extract_words(self, text:str) -> list[str]:
        return [word.lower() for word in re.findall(r'\w+', text)]
    
    def _build_tf_idf(self):
        # extract all words in the corpus using regex
        corpus_words = [self._extract_words(text) for text in self.corpus]
        
        # compute term frequency per each document
        self.tf = []
        for doc_words in corpus_words:
            doc_tf = {}
            for word in doc_words:
                doc_tf[word] = doc_tf.get(word, 0) + 1
            self.tf.append(doc_tf)

        # compute inverse document frequency for each word
        self.idf = {}
        for doc_tf in self.tf:
            for word in doc_tf:
                self.idf[word] = self.idf.get(word, 0) + 1
        for word in self.idf:
            self.idf[word] = len(self.tf) / self.idf[word]

        # compute tf-idf for each word in each document
        self.tf_idf = []
        for doc_tf in self.tf:
            doc_tf_idf = {}
            for word in doc_tf:
                doc_tf_idf[word] = doc_tf[word] * self.idf[word]
            self.tf_idf.append(doc_tf_idf)


    def search(self, query:str) -> list[tuple[str, float]]:
        # extract words from the query
        query_words = self._extract_words(query)
        
        # compute tf-idf for the query
        results = []
        for i, doc_tf_idf in enumerate(self.tf_idf):
            score = 0
            for word in query_words:
                score += doc_tf_idf.get(word, 0)
            if score > 0:
                results.append((self.corpus[i], score))
        
        results.sort(key=lambda x: x[1], reverse=True)

        return results


class NeuralSearch(TF_IDF):...


if __name__ == '__main__':
    main()