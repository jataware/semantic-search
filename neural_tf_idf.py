import json
from transformers import BertTokenizer, BertModel, logging
from abc import ABC, abstractmethod
import re
import torch
from tqdm import tqdm

import pdb

def main():

    with open('indicators.jsonl') as f:
        lines = f.readlines()
        indicators = [json.loads(line) for line in lines]
    
    descriptions = [indicator['_source']['description'] for indicator in indicators]
    
    
    text_search = PlaintextSearch(descriptions)
    neural_search = NeuralSearch(descriptions)
    
    while True:
        query = input('Search: ')
        text_results = text_search.search(query)
        neural_results = neural_search.search(query)
        print_results(text_results, 'text')
        print_results(neural_results, 'neural')
    


    




def print_results(results:list[tuple[str,float]], search_type:str):
    print(f'--------------------------------- {search_type} results: ---------------------------------')
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
        for doc, doc_tf_idf in zip(self.corpus, self.tf_idf):
            score = 0
            for word in query_words:
                score += doc_tf_idf.get(word, 0)
            if score > 0:
                results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)

        return results


class NeuralSearch(TF_IDF):
    """neural TF-IDF search based on BERT"""
    def __init__(self, corpus:list[str]):

        # load BERT tokenizer and model from HuggingFace
        with torch.no_grad():
            logging.set_verbosity_error()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')

            # move model to GPU
            self.model = self.model.cuda()

        # set up the corpus and compute tf-idf
        self.corpus = corpus
        self._build_tf_idf()

    def _build_tf_idf(self):
        with torch.no_grad():
            
            # convert each document to a BERT token embedding
            tokenized_corpus = self.tokenizer(self.corpus, return_tensors='pt', padding='max_length', truncation=True)

            # break the data into chunks, and move to GPU
            chunk_size = 10
            tokenized_corpus_chunks = []
            for i in range(0, len(tokenized_corpus['input_ids']), chunk_size):
                tokenized_corpus_chunks.append({k: v[i:i+chunk_size].cuda() for k, v in tokenized_corpus.items()})
            
            # encode each document using BERT
            encoded_corpus_chunks = []
            for chunk in tqdm(tokenized_corpus_chunks, desc='neural encoding corpus'):
                encoded_corpus_chunks.append(self.model(**chunk).last_hidden_state)
            
            self.encoded_corpus = torch.cat(encoded_corpus_chunks, dim=0)


    def search(self, query:str) -> list[tuple[str, float]]:
        with torch.no_grad():
            # tokenize the query, and encode with BERT
            encoded_query = self.tokenizer(query, return_tensors='pt')
            encoded_query = {k: v.cuda() for k, v in encoded_query.items()} # move to GPU
            encoded_query = self.model(**encoded_query).last_hidden_state

            # compute cosine similarity between each vector in the query to each vector in each document
            results = []
            for doc, encoded_doc in zip(self.corpus, self.encoded_corpus):
                scores = torch.cosine_similarity(encoded_query, encoded_doc[:,None], dim=2)
                score = scores.max(dim=0).values.mean().item()
                
                if score > 0:
                    results.append((doc, score))
                
               
            results.sort(key=lambda x: x[1], reverse=True)

            return results[:10] #DEBUG: only return the top 10 results




if __name__ == '__main__':
    main()