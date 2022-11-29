# simple place to collect all the corpora available to search over

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Union, Hashable

import pdb

@dataclass
class Corpus:
    docs: Union[list[str], dict[Hashable, str]]
    
    def get_keyed_corpus(self):
        if isinstance(self.docs, dict):
            return self.docs
        else:
            return {i: doc for i, doc in enumerate(self.docs)}

    def get_unkeyed_corpus(self):
        if isinstance(self.docs, list):
            return self.docs
        else:
            return list(self.docs.values())

    def __post_init__(self):
        """some validation on the input data to ensure it's in the right format"""
        assert isinstance(self.docs, list) or isinstance(self.docs, dict), 'docs must be a list or a dict'
        if isinstance(self.docs, dict):
            assert all(isinstance(key, Hashable) for key in self.docs.keys()), 'keys must be hashable'
            assert all(isinstance(doc, str) for doc in self.docs.values()), 'docs must be strings'
        else:
            assert all(isinstance(doc, str) for doc in self.docs), 'docs must be strings'
        



#TODO: instead of any, tuple version should take a key type (i.e. hashable)
class CorpusLoader(ABC):
    @staticmethod
    @abstractmethod
    def get_corpus() -> Corpus: ...
    #TODO: maybe some way to validate documents should be less than 512 tokens...


class ResearchPapers(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus:
        
        docs = {}
        with open('data/dart_cdr.json_mar_2022') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                doc = json.loads(line)['extracted_text']
                chunks = ResearchPapers.chunk_doc(doc)
                for j, chunk in enumerate(chunks):
                    docs[f'{i}_{j}'] = chunk
        
        return Corpus(docs)
    
    
    @staticmethod
    def chunk_doc(doc:str) -> list[str]:
        """split the document on paragraphs (separated by newlines)"""
        paragraphs = [*filter(len, doc.split('\n'))] #remove empty paragraphs
        return paragraphs
        
        




