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



# def preprocess_data():
#     # input_datapath = 'data/fine_food_reviews_1k.csv'  # to save space, we provide a pre-filtered dataset
#     input_datapath = 'data/Reviews.csv'
#     df = pd.read_csv(input_datapath, index_col=0)
#     df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]
#     df = df.dropna()
#     df['combined'] = "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
#     df.head(2)

#     # subsample to 1k most recent reviews and remove samples that are too long
#     df = df.sort_values('Time').tail(1_100)
#     df.drop('Time', axis=1, inplace=True)

#     tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

#     # remove reviews that are too long
#     df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))
#     df = df[df.n_tokens<2000].tail(1_000)
#     len(df)


#     # This will take just under 10 minutes
#     print("Getting embeddings. This will take about 10 minutes...")
#     df['babbage_similarity'] = df.combined.apply(lambda x: get_embedding(x, engine='text-similarity-babbage-001'))
#     df['babbage_search'] = df.combined.apply(lambda x: get_embedding(x, engine='text-search-babbage-doc-001'))
#     df.to_csv('data/fine_food_reviews_with_embeddings_1k.csv')




# def main():
#     df = get_search_df()
#     res = search_reviews(df, "delicious beans", n=3)
#     pdb.set_trace()
#     pass

# def get_search_df() -> pd.DataFrame:
#     datafile_path = "https://cdn.openai.com/API/examples/data/fine_food_reviews_with_embeddings_1k.csv"  # for your convenience, we precomputed the embeddings
#     df = pd.read_csv(datafile_path)
#     df["babbage_search"] = df.babbage_search.apply(eval).apply(np.array)

#     return df


# # search through the reviews for a specific product
# def search_reviews(df, product_description, n=3, pprint=True):
#     embedding = get_embedding(
#         product_description,
#         engine="text-search-babbage-query-001"
#     )
#     df["similarities"] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))

#     res = (
#         df.sort_values("similarities", ascending=False)
#         .head(n)
#         .combined.str.replace("Title: ", "")
#         .str.replace("; Content:", ": ")
#     )
#     if pprint:
#         for r in res:
#             print(r[:200])
#             print()
#     return res





# if __name__ == '__main__':
#     set_api_key()
#     main()