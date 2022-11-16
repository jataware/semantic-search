from transformers import BertTokenizer, BertModel, logging
import torch
from tqdm import tqdm

from search import Search


class BertSearch(Search):
    """neural TF-IDF search based on BERT"""
    def __init__(self, corpus:list[str], model='bert-base-uncased'):

        # load BERT tokenizer and model from HuggingFace
        with torch.no_grad():
            logging.set_verbosity_error()
            self.tokenizer = BertTokenizer.from_pretrained(model)
            self.model = BertModel.from_pretrained(model)

            # move model to GPU
            self.model = self.model.cuda()

        # set up the corpus and compute tf-idf
        self.corpus = corpus
        self._build_tf_idf()

    def _build_tf_idf(self, save_path='data/bert_encoded_corpus.pt'):
        #try to load the encoded corpus from disk
        try:
            self.encoded_corpus = torch.load(save_path)
            print('Loaded bert encoded corpus from disk')
            return
        except FileNotFoundError:
            pass

        print('encoding corpus with BERT')
        with torch.no_grad():
            
            # convert each document to a BERT token embedding
            tokenized_corpus = self.tokenizer(self.corpus, return_tensors='pt', padding='max_length', truncation=True)

            # break the data into chunks, and move to GPU
            chunk_size = 20
            tokenized_corpus_chunks = []
            for i in range(0, len(tokenized_corpus['input_ids']), chunk_size):
                tokenized_corpus_chunks.append({k: v[i:i+chunk_size].cuda() for k, v in tokenized_corpus.items()})
            
            # encode each document using BERT
            encoded_corpus_chunks = []
            for chunk in tqdm(tokenized_corpus_chunks, desc='encoding corpus with BERT'):
                encoded_corpus_chunks.append(self.model(**chunk).last_hidden_state)
            
            self.encoded_corpus = torch.cat(encoded_corpus_chunks, dim=0)

            #save the corpus to disk
            torch.save(self.encoded_corpus, save_path)


    def search(self, query:str, n:int=None) -> list[tuple[str, float]]:
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

            if n is not None:
                results = results[:n]

            return results

