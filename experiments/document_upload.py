"""
prereqs/dependencies:

sudo apt install ghostscript
sudo apt install tesseract-ocr

pip install ocrmypdf
pip install pypdf
pip install numpy
pip install torch
pip install transformers
pip install sentence-transformers
"""



import os
from glob import glob
from pypdf import PdfReader
import ocrmypdf
from typing import Generator
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import logging

import pdb


def get_metadata(path):
    reader = PdfReader(path)
    meta = reader.metadata

    pages = len(reader.pages)

    author = meta.author
    if is_blacklisted_author(author):
        author = None

    title = meta.title

    try:
        creation_date = meta.creation_date
    except:
        creation_date = None

    subject = meta.subject #most of the time, this is empty

    return (pages, author, title, creation_date, subject)



def extract_text(path: str) -> list[tuple[str, int]]:
    work_path = 'tmp.pdf'
    ocrmypdf.ocr(path, work_path, language='eng', progress_bar=False, redo_ocr=True, sidecar='tmp.txt')
    reader = PdfReader(work_path)
    pages = [page.extract_text() for page in reader.pages]

    #create a map from the line number to the page number
    num_lines = np.array([len(page.splitlines()) for page in pages] + [float(9999999)]) # lines on each page
    cumulative_line_counts = np.cumsum(num_lines) # line number to page number
    line_to_page = lambda line_num: np.argmax(cumulative_line_counts >= line_num)

    text = '\n'.join(pages)
    
    #combine any adjacent lines with more than 5 words (hacky way to combine paragraphs)
    lines = text.splitlines()
    paragraphs = []
    paragraph = []
    for line_number, line in enumerate(lines):
        page_number = line_to_page(line_number)
        line = line.strip()
        if 5 < len(line.split()):# < 50: # if the line has more than 5 words, but less than 100, it's probably a line of a larger paragraph
            paragraph.append((line, page_number))
        else:
            paragraph = [p for p in paragraph if p[0]] # filter out empty strings
            paragraph_txt = ' '.join([p[0] for p in paragraph])
            paragraph_pages = {p[1] for p in paragraph}
            if paragraph_txt:
                paragraphs.append((paragraph_txt, paragraph_pages))
            if line:
                paragraphs.append((line, {page_number}))
            paragraph = []

    if paragraph:
        paragraph_txt = ' '.join([p[0] for p in paragraph])
        paragraph_pages = {p[1] for p in paragraph}
        if paragraph_txt:
            paragraphs.append((paragraph_txt, paragraph_pages))

    
    #TODO: look into extra filtering for cleaning up the extracted text
    #      this section didn't work very well... 
    # #extra filtering
    # def filter_punctuation(paragraph: str, punctuation: str) -> str:
    #     joiner = f'{punctuation} ' if punctuation != ' ' else ' '
    #     chunks = paragraph.split(punctuation)
    #     chunks = [c.strip() for c in chunks]
    #     chunks = [c for c in chunks if c]
    #     paragraph = joiner.join(chunks)
    #     return paragraph

    # filtered_paragraphs = []
    # for paragraph, pages in paragraphs:
    #     # filtered_paragraph = filter_punctuation(paragraph, '.')
    #     filtered_paragraph = filter_punctuation(paragraph, ' ')
    #     #TODO: other filters?
    #     if filtered_paragraph:
    #         filtered_paragraphs.append((filtered_paragraph, pages))
        
    # paragraphs = filtered_paragraphs



    #take the smallest page number from the pages that the paragraph is on
    paragraphs = [(paragraph, min(pages)) for paragraph, pages in paragraphs]
    

    return paragraphs



class Embedder:
    """
    Convert a list of strings to embeddings

    Example:
    ```
    model = Embedder(cuda=True)
    sentences = ['this is a sentence', 'this is another sentence', 'this is a third sentence']
    embeddings = model.embed(sentences)
    ```

    """
    def __init__(self, *, model='all-mpnet-base-v2', cuda=True, batch_size=32):

        self.batch_size = batch_size
        
        #create an instance of the model, and optionally move it to GPU
        with torch.no_grad():
            logging.set_verbosity_error()
            self.model = SentenceTransformer(model)
            if cuda:
                self.model = self.model.cuda()

    def embed(self, sentences: list[str]) -> list[np.ndarray]:
        """
        embed a list of sentences
        """
        with torch.no_grad():
            embeddings = self.model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False)
        embeddings = [e for e in embeddings] #convert to list
        return embeddings



authors_blacklist = {
    'user', 
    'utente di', #`user of` in italian 
    'microsoft',
    'office',
    # 'BANTIV', #what is this?
    'adobe',
    'acrobat',
}

def is_blacklisted_author(author: str) -> bool:
    """check if an author string is made up of blacklisted words (indicating no actual author given)"""
    if not author:
        return True
    if any([a.strip() in authors_blacklist for a in author.lower().split()]):
        return True
    return False





def get_pdfs(root: str) -> Generator[str, None, None]:
    """
    get all pdf files in the root directory and its subdirectories
    """
    for path in glob(os.path.join(root, '**', '*.pdf'), recursive=True):
        yield path



def get_authors(root) -> set[str]:
    """
    simple helper function for getting all authors from the pdfs in a directory. 
    Mainly used for generating the author blacklist.
    """
    authors = set()
    for path in get_pdfs(root):
        reader = PdfReader(path)
        meta = reader.metadata
        if meta.author:
            authors.add(meta.author)
    return authors





if __name__ == '__main__':
    
    model = Embedder(cuda=True)

    # get all pdf files in the root directory and its subdirectories
    for path in get_pdfs('data/transition_reports'):
        metadata = get_metadata(path)
        pages, author, title, creation_date, subject = metadata
        print('--------metadata--------')
        print(f'path: {path}')
        print(f'pages: {pages}')
        print(f'author: {author}')
        print(f'title: {title}')
        print(f'creation date: {creation_date}')
        print(f'subject: {subject}')
        print('---------------------------------------------------------')
        paragraphs = extract_text(path)
        text = '\n'.join([f'(page {p[1]}) ' + p[0] for p in paragraphs])
        print(text)
        print('\n\n')

        # embeddings = model.embed([p for p, _ in paragraphs])
        # pdb.set_trace()
        # break
