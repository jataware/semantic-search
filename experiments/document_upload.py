#read a pdf document, and extract the text, author, publisher, date, etc. from it

#process:
# 1. try to read the pdf text directly by parsing the file
# 2. any images should have ocr run on them
#    -> same if all pages are pure images

"""
to use ocrmypdf, you need to install ghostscript and tesseract-ocr
sudo apt install ghostscript
sudo apt install tesseract-ocr
"""

import os
from glob import glob
from pypdf import PdfReader
import ocrmypdf
from typing import Generator
import re
import numpy as np


import pdb


authors_blacklist = {
    'user', 
    'utente di', #`user of` in italian 
    'microsoft',
    'office',
    # 'BANTIV', #what is this?
    'adobe',
    'acrobat',
}


def get_pdfs(root: str) -> Generator[str, None, None]:
    """
    get all pdf files in the root directory and its subdirectories
    """
    for path in glob(os.path.join(root, '**', '*.pdf'), recursive=True):
        yield path


def convert_pdf(path: str, skip_ocr=False) -> str:
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

    #collect metadata
    meta = reader.metadata
    author = meta.author
    if author in authors_blacklist:
        author = None
    title = meta.title
    try:
        creation_date = meta.creation_date
    except:
        creation_date = None

    metadata = (author, title, creation_date)

    return paragraphs, metadata

# def read_pdf(path):
#     reader = PdfReader(path)

#     meta = reader.metadata

#     print(f'pages: {len(reader.pages)}')

#     # All of the following could be None!
#     print(f'path: {path}')
#     print(f'author: {meta.author}')
#     print(f'title: {meta.title}')
#     try:
#         print(f'creatione date: {meta.creation_date}')
#     except:
#         print('creation date: None')
#     # print(meta.creator)
#     # print(meta.producer)
#     # print(meta.subject)
#     # print('\n\n')

#     # pdb.set_trace()

def get_authors(root) -> set[str]:
    authors = set()
    for path in get_pdfs(root):
        reader = PdfReader(path)
        meta = reader.metadata
        if meta.author:
            authors.add(meta.author)
    return authors

if __name__ == '__main__':
    # authors = get_authors('data/transition_reports')
    # pdb.set_trace()
    
    
    
    results = {}
    # get all pdf files in the root directory and its subdirectories
    for i,path in enumerate(get_pdfs('data/transition_reports')):
        paragraphs, metadata = convert_pdf(path)
        text = '\n'.join([f'(page {list(p[1])}) ' + p[0] for p in paragraphs])
        results[path] = text
        print('---------------------------------------------------------')
        print(path)
        print(text)
        print('--------metadata--------')
        (author, title, creation_date) = metadata
        print(f'author: {author}')
        print(f'title: {title}')
        print(f'creation date: {creation_date}')
        print('\n\n')
        # break
        # if i >= 2:
        #     break

    
    # ocr_pdf('data/test_pdfs/20230123111924_001.pdf')
