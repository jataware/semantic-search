import json
from .corpora import Corpus, CorpusLoader

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
        
        