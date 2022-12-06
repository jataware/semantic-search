# run in root folder: `python -m experiments.uaz_documents_comparison`
import pandas as pd
from search.corpora import ResearchPapers
import json

import pdb

def main():
    # corpus = ResearchPapers.get_corpus()
    # print('num documents:',len(corpus))
    count_uaz_concepts_in_docs()


def count_uaz_concepts_in_docs():
    concepts = set()
    # docs = {}
    with open('data/dart_cdr.json_mar_2022') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = json.loads(line)
            for i in line['annotations']:
                if i['label'] == 'qntfy-categories-annotator':
                    for concept_dict in i['content']:
                        concepts.add(concept_dict['value'])
                # print(f"type: {i['type']}")
                # pdb.set_trace()
            # print(line['annotations'])
            # line['annotations']
            # try:
            #     doc = line['extracted_text']
            #     n_lines += 1
            #     chunks = ResearchPapers.chunk_doc(doc)
            #     for j, chunk in enumerate(chunks):
            #         docs[f'{i}_{j}'] = chunk
            # except:
            #     pdb.set_trace()
            #     pass
    print(f'concepts: {len(concepts)}')
    print(concepts)

if __name__ == '__main__':
    main()