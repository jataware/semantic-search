# simple place to collect all the corpora available to search over

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterator, Iterable


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

    def __len__(self) -> int:
        return len(self.keyed_corpus)

    def __iter__(self) -> Iterator[T]:
        return iter(self.keyed_corpus)

    def keys(self) -> Iterable[T]:
        return self.keyed_corpus.keys()

    def values(self) -> Iterable[str]:
        return self.keyed_corpus.values()

    def items(self) -> Iterable[tuple[T, str]]:
        return self.keyed_corpus.items()

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




# TODO: maybe chunk on a sentence level rather than a paragraph level
class DartPapers(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus[tuple[int,int]]:  
        
        with open('data/dart_cdr.json_mar_2022') as f:
            lines = f.readlines()

        docs = {}
        for i, line in enumerate(lines):
            doc = json.loads(line)
            id = doc['document_id']
            try:
                text = doc['extracted_text']
            except:
                print(f'error parsing document {id} on line {i}. No extracted text.')
                continue
            
            chunks = DartPapers.chunk_doc(text)
            for j, chunk in enumerate(chunks):
                docs[(id,j)] = chunk
        
        return Corpus(docs)
    
    
    @staticmethod
    def chunk_doc(doc:str) -> list[str]:
        """split the document on paragraphs (separated by newlines)"""
        paragraphs = [*filter(len, doc.split('\n'))] #remove empty paragraphs
        return paragraphs
        
        