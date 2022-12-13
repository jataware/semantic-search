import json
from .corpora import Corpus, CorpusLoader

# TODO: maybe chunk on a sentence level rather than a paragraph level
class DartPapers(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus[str]:  
        
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
            
            docs[id] = text

        return Corpus(docs)

    @staticmethod
    def get_paragraph_corpus() -> Corpus[tuple[str,int]]:
        corpus = DartPapers.get_corpus()
        return Corpus.chunk(corpus, DartPapers.chunk_paragraphs)

    @staticmethod
    def get_sentence_corpus() -> Corpus[tuple[str,int,int]]:
        corpus = DartPapers.get_corpus()
        return Corpus.chunk(corpus, DartPapers.chunk_sentences)

    
    @staticmethod
    def chunk_paragraphs(doc:str) -> list[str]:
        """split the document on paragraphs (separated by newlines)"""
        paragraphs = [*filter(len, doc.split('\n'))] #remove empty paragraphs
        return paragraphs
        
    #TODO: how to handle abbreviations, other periods that aren't sentence endings?
    @staticmethod
    def chunk_sentences(doc:str) -> list[str]:
        """split the document on sentences (separated by periods)"""
        raise NotImplementedError