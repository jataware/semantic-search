# run in root folder: `python -m experiments.uaz_documents_comparison`
import pandas as pd
from data.dart_papers import DartPapers
import json

import pdb

def main():
    # print('num documents:',len(corpus))
    concept_map = get_uaz_concepts_to_docs()


def get_concepts(actor: dict) -> list[str]:
    results = [i['name'] for i in actor['concept']['db_refs']['WM_FLAT']]
    return results

def get_docs(evidence: list) -> list[str]:
    assert isinstance(evidence, list), f'expected list, got {type(evidence)}'
    doc_ids = []
    for e in evidence:
        try:
            if e['source_api'] == 'eidos':
                doc_ids.append(e['text_refs']['DART'])
        except:
            pdb.set_trace()
    return doc_ids

def get_uaz_concepts_to_docs():
    corpus = DartPapers.get_corpus()
    doc_ids = set(id for id, _ in corpus.keys())

    with open('data/statements_2022_march_v4.jsonl') as f:
        lines = f.readlines()
    
    concepts = {} # concept -> set<doc-id>
    docs = set() # list of all docs encountered

    for line in lines:
        data = json.loads(line)
        line_concepts = get_concepts(data['subj']) + get_concepts(data['obj'])
        doc_ids = get_docs(data['evidence'])
        for concept in line_concepts:
            if concept not in concepts:
                concepts[concept] = set()
            concepts[concept].update(doc_ids)
        docs.update(doc_ids)

    print(f'concepts: {concepts}')
    pdb.set_trace()    
    
    
    # # docs = {}
    # with open('data/dart_cdr.json_mar_2022') as f:
    #     lines = f.readlines()
    
    # for i, line in enumerate(lines):
    #     line = json.loads(line)
    #     for i in line['annotations']:
    #         if i['label'] == 'qntfy-categories-annotator':
    #             for concept_dict in i['content']:
    #                 concepts.add(concept_dict['value'])
    #         # print(f"type: {i['type']}")
    #         # pdb.set_trace()
    #     # print(line['annotations'])
    #     # line['annotations']
    #     # try:
    #     #     doc = line['extracted_text']
    #     #     n_lines += 1
    #     #     chunks = ResearchPapers.chunk_doc(doc)
    #     #     for j, chunk in enumerate(chunks):
    #     #         docs[f'{i}_{j}'] = chunk
    #     # except:
    #     #     pdb.set_trace()
    #     #     pass
    # print(f'concepts: {len(concepts)}')
    # print(concepts)

if __name__ == '__main__':
    main()