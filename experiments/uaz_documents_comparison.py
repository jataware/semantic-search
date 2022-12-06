# run in root folder: `python -m experiments.uaz_documents_comparison`
import pandas as pd
from data.dart_papers import DartPapers
import json

import pdb

def main():
    # print('num documents:',len(corpus))
    concept_map = get_uaz_concepts_to_docs()


def get_uaz_concepts_to_docs():
    corpus = DartPapers.get_corpus()
    doc_ids = set(id for id, _ in corpus.keys())

    with open('data/statements_2022_march_v4.jsonl') as f:
        lines = f.readlines()
    
    for line in lines:
        data = json.loads(line)
        for evidence in data['evidence']: # [<index>]['text_refs']['DART']
            id = evidence['text_refs']['DART']
            pdb.set_trace()
        # id = data['id']
        pdb.set_trace()
    
    
    pdb.set_trace()
    
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