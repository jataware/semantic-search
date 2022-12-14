# run in root folder: `python -m experiments.uaz_documents_comparison`
import pandas as pd
from data.dart_papers import DartPapers
from data.wm_ontology import FlatOntology
from data.corpora import Corpus, CorpusLoader
from search.bert_search import BertSentenceSearch
import json
from tqdm import tqdm

import pdb


#TODO:
# - basic term to documents matching: check if top 20 from this in the uaz matches
# - basic term matching: make matches out of top 20 from this, checking for paragraphs that match multiple terms, compare to uaz matches
# - tbd how to do basic polarity on pairs of matched terms...



class DartTop100(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus[str]:
        docs_corpus = DartPapers.get_corpus()
        ontology = FlatOntology.get_corpus()
        concept_map = get_uaz_concepts_to_docs(filter_empty=True)

        found_concepts = []
        for concept in concept_map:
            concept = concept.replace(' ', '_')
            if concept in ontology:
                found_concepts.append(concept)

        #collect the top 100 most frequent papers
        paper_counts = {}
        for concept in found_concepts:
            if concept not in concept_map:
                continue
            for paper in concept_map[concept]:
                if paper not in paper_counts:
                    paper_counts[paper] = 0
                paper_counts[paper] += 1

        top_papers = sorted(paper_counts.items(), key=lambda x: x[1], reverse=True)[:100]

        docs = {doc_id: docs_corpus[doc_id] for doc_id, score in top_papers}
 
        return Corpus.chunk(docs, DartPapers.chunk_paragraphs)





# from scalene import scalene_profiler
def main():
    corpus = DartPapers.get_paragraph_corpus()
    ontology = FlatOntology.get_corpus()


    #blacklist function, reject results with less than 100 alphabetical characters
    import re, string
    pattern = re.compile('[^a-zA-Z]')
    blacklist = lambda x: len(x) < 100 or len(pattern.sub('', x)) < 100
    
    engine = BertSentenceSearch(corpus, save_name=DartPapers.__name__, batch_size=256, blacklist=blacklist)
    
    # query = ontology['food']

    for key, query in ontology.items():
        # query = input('Enter query: ')
        print("-----------------------------------------------------")
        input(f'press ENTER to search query for ({key}) "{query}"')

        matches = engine.search(query, n=10)
        # print the results of the search
        print('Top 10 matches:')
        for match_id, score in matches:
            raw_text = corpus[match_id]   # get the matching text for the given id
            print(raw_text, end='\n\n\n') # print the text

    # pdb.set_trace()
    # concept_map = get_uaz_concepts_to_docs(filter_empty=True)

    # found_concepts = []
    # for concept in concept_map:
    #     concept = concept.replace(' ', '_')
    #     if concept in ontology:
    #         found_concepts.append(concept)

    # #collect the top 100 most frequent papers
    # paper_counts = {}
    # for concept in found_concepts:
    #     if concept not in concept_map:
    #         continue
    #     for paper in concept_map[concept]:
    #         if paper not in paper_counts:
    #             paper_counts[paper] = 0
    #         paper_counts[paper] += 1

    # top_papers = sorted(paper_counts.items(), key=lambda x: x[1], reverse=True)[:100]

    # corpus = DartTop100.get_corpus()
    pdb.set_trace()


    #do a search for each concept over the corpus
    # for concept in found_concepts:
    #     results = engine.search(ontology[concept],n=3)


def main2():
    corpus = DartPapers.get_paragraph_corpus()
    ontology = FlatOntology.get_corpus()


    #blacklist function, reject results with less than 100 alphabetical characters
    import re, string
    pattern = re.compile('[^a-zA-Z]')
    blacklist = lambda x: len(x) < 1000 or len(pattern.sub('', x)) < 1000
    
    engine = BertSentenceSearch(corpus, save_name=DartPapers.__name__, batch_size=256, blacklist=blacklist)

    links = {} # map<paragraph_id, list<concept_id>>


    for key, query in tqdm([*ontology.items()]):
        # query = input('Enter query: ')
        # print("-----------------------------------------------------")
        # input(f'press ENTER to search query for ({key}) "{query}"')

        matches = engine.search(query, n=10)
        for match_id, score in matches:
            if match_id not in links:
                links[match_id] = []
            links[match_id].append(key)

        # # print the results of the search
        # print('Top 10 matches:')
        # for match_id, score in matches:
        #     raw_text = corpus[match_id]   # get the matching text for the given id
        #     print(raw_text, end='\n\n\n') # print the text


    for paragraph_id, concept_ids in links.items():
        if len(concept_ids) > 1:
            # print(paragraph_id, concept_ids)
            print('-------------------------')
            print(f'{corpus[paragraph_id]}')
            for concept_id in concept_ids:
                print(f'- {concept_id}: {ontology[concept_id]}')

            print('\n\n\n')
    
    pdb.set_trace()


def get_concepts(actor: dict) -> list[str]:
    results = [i['name'] for i in actor['concept']['db_refs']['WM_FLAT']]
    return results

def get_docs(evidence: list) -> list[str]:
    assert isinstance(evidence, list), f'expected list, got {type(evidence)}'
    doc_ids = []
    for e in evidence:
        if e['source_api'] == 'eidos':
            doc_ids.append(e['text_refs']['DART'])
    return doc_ids

def get_uaz_concepts_to_docs(*, filter_empty=False):

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


    # filter out all concepts that don't have papers
    if filter_empty:
        concepts = {k: v for k, v in concepts.items() if len(v) >= 1}
    
    return concepts
    
    
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
    # main()
    main2()