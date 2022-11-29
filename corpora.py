# simple place to collect all the corpora available to search over

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Union, Any, Hashable

import pdb

@dataclass
class Corpus:
    docs: Union[list[str], dict[Hashable, str]]
    
    def get_keyed_corpus(self):
        if all(isinstance(doc, tuple) for doc in self.docs):
            return self.docs
        elif all(isinstance(doc, str) for doc in self.docs):
            return [(i, doc) for i, doc in enumerate(self.docs)]
        else:
            raise TypeError('docs must be a list of strings or list of tuples') #TODO: make this check run at init

    def __post_init__(self):
        assert isinstance(self.docs, list), 'docs must be a list'
        
        #check that all the docs are strings or tuples
        pdb.set_trace()


#TODO: instead of any, tuple version should take a key type (i.e. hashable)
class CorpusLoader(ABC):
    @staticmethod
    @abstractmethod
    def get_corpus() -> Corpus: ...
    #TODO: maybe some way to validate documents should be less than 512 tokens...


class ResearchPapers(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus:
        
        docs = []
        with open('data/sample_docs.jsonl') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                doc = json.loads(line)['extracted_text']
                chunks = ResearchPapers.chunk_doc(doc)
                for j, chunk in enumerate(chunks):
                    docs.append((f'{i}_{j}', chunk))
        
        return Corpus(docs)
    
    
    @staticmethod
    def chunk_doc(doc:str) -> list[str]:
        """split the document on paragraphs (separated by newlines)"""
        paragraphs = [*filter(len, doc.split('\n'))] #remove empty paragraphs
        return paragraphs
        
        




