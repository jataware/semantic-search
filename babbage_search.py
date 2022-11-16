import os
import numpy as np
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
from search_types import Search


import pdb



class BabbageSearch(Search):
    def __init__(self, corpus: list[str]):
        set_api_key()
        self.corpus = corpus
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.embeddings = self._build_embeddings()

    def _build_embeddings(self, save_path='data/babbage_encoded_corpus.npy'):
        #check if a saved version of the embeddings exists
        try:
            embeddings = np.load(save_path)
            print('Loaded babbage encoded corpus from disk')
            return embeddings
        except FileNotFoundError:
            pass
                    
        print("Building embeddings for babbage search")
        df = pd.DataFrame(self.corpus, columns=["text"])
        df['search'] = df['text'].apply(lambda x: get_embedding(x, engine='text-search-babbage-doc-001'))

        embeddings = np.array(df['search'].tolist())
        np.save(save_path, embeddings)
        return embeddings


    def search(self, query: str, n: int = None) -> list[tuple[str, float]]:
        encoded_query = get_embedding(query, engine='text-search-babbage-doc-001')
        encoded_query = np.array(encoded_query)#.reshape(1, -1)
        

        results = []
        for doc, encoded_doc in zip(self.corpus, self.embeddings):
            score = cosine_similarity(encoded_query, encoded_doc)
            
            if score > 0:
                results.append((doc, score))
            
            
        results.sort(key=lambda x: x[1], reverse=True)

        if n is not None:
            results = results[:n]

        return results
    
        
def set_api_key():
    openai.organization = "org-x0wb7zqe7vQpjdKjNom7KfFh"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    #test the api key
    openai.Model.list()


