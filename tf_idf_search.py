import re
from search import Search


class PlaintextSearch(Search):
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


    def search(self, query:str, n:int=None) -> list[tuple[str, float]]:
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

        if n is not None:
            results = results[:n]

        return results
