# simple place to collect all the corpora available to search over

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Union, Any, TypeVar, Generic

import pdb

#TODO: figure out how to get mypy to allow int/string to work with Hashable: https://github.com/python/mypy/issues/2412
# i.e. we want to be able to say dict[Hashable, string], where Hashable could be an int, string, etc.
#TODO: replace optional passing in of pure list with a helper method that takes a list. the init should only take a dict


T = TypeVar('T')

class Corpus(Generic[T]):
    def __init__(self, docs: dict[T, str]):
        assert isinstance(docs, dict), "corpus must be a dict[T, str]"
        assert all(isinstance(doc, str) for doc in docs.values()), 'corpus may only contain strings'
        self.keyed_corpus = docs

    def get_keyed_corpus(self) -> dict[T, str]:
        return self.keyed_corpus

    def __getitem__(self, key: T) -> str:
        return self.keyed_corpus[key]

    @staticmethod
    def from_list(docs: list[str]) -> 'Corpus[int]':
        return Corpus({i: doc for i, doc in enumerate(docs)})

    @staticmethod
    def from_dict(docs: dict[T, str]) -> 'Corpus[T]':
        return Corpus(docs)


#TODO: instead of any, tuple version should take a key type (i.e. hashable)
class CorpusLoader(ABC):
    @staticmethod
    @abstractmethod
    def get_corpus() -> Corpus: ...
    #TODO: maybe some way to validate documents should be less than 512 tokens...


class ResearchPapers(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus[str]:
        
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
        
        


class Indicators(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus[int]:
        
        with open('data/indicators.jsonl') as f:
            lines = f.readlines()
            indicators = [json.loads(line) for line in lines]

        descriptions = []
        for indicator in indicators:
            for out in indicator['_source']['outputs']:
                #display name, description, unit, unit description
                description = \
f"""name: {out['name']};
display name: {out['display_name']};
description: {out['description']};
unit: {out['unit']};
unit description: {out['unit_description']};"""
                descriptions.append(description)

        docs = {i: doc for i, doc in enumerate(descriptions)}

        return Corpus(docs)
