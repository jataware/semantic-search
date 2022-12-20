# run in root folder: `python -m experiments.uaz_documents_comparison`
import pandas as pd
from data.dart_papers import DartPapers
from data.wm_ontology import FlatOntology
from data.corpora import Corpus, CorpusLoader
from search.bert_search import BertSentenceSearch
import json
from tqdm import tqdm
import torch
import re

import pdb


#TODO:
# - basic term to documents matching: check if top 20 from this in the uaz matches
# - basic term matching: make matches out of top 20 from this, checking for paragraphs that match multiple terms, compare to uaz matches
# - tbd how to do basic polarity on pairs of matched terms...



leaf_nodes = FlatOntology.get_leaf_nodes()
blacklisted_nodes = set(FlatOntology.get_blacklisted_nodes())
ontology = FlatOntology.get_corpus()

def valid_ontology():
    for key in leaf_nodes:
        if key in blacklisted_nodes:
            continue
        yield key, ontology[key]


#blacklist function, reject results with less than some number of alphabetical characters
pattern = re.compile('[^a-zA-Z]')
def blacklist_doc(N=500):
    return lambda x: len(x) < N or len(pattern.sub('', x)) < N




# class DartTop100(CorpusLoader):
#     @staticmethod
#     def get_corpus() -> Corpus[str]:
#         docs_corpus = DartPapers.get_corpus()
#         ontology = FlatOntology.get_corpus()
#         concept_map = get_uaz_concepts_to_docs(filter_empty=True)

#         found_concepts = []
#         for concept in concept_map:
#             concept = concept.replace(' ', '_')
#             if concept in ontology:
#                 found_concepts.append(concept)

#         #collect the top 100 most frequent papers
#         paper_counts = {}
#         for concept in found_concepts:
#             if concept not in concept_map:
#                 continue
#             for paper in concept_map[concept]:
#                 if paper not in paper_counts:
#                     paper_counts[paper] = 0
#                 paper_counts[paper] += 1

#         top_papers = sorted(paper_counts.items(), key=lambda x: x[1], reverse=True)[:100]

#         docs = {doc_id: docs_corpus[doc_id] for doc_id, score in top_papers}
 
#         return Corpus.chunk(docs, DartPapers.chunk_paragraphs)


def get_paragraph_terms(corpus=None, engine=None, n=10):
    """return a map from paragraph id to the set of terms that matched that paragraph"""
    if corpus is None:
        corpus = DartPapers.get_paragraph_corpus()
    if engine is None:
        engine = BertSentenceSearch(corpus, save_name=DartPapers.__name__, batch_size=256, blacklist=blacklist_doc())

    matches = {}
    for key, query in tqdm([*valid_ontology()], desc='matching concept pairs'):
        for match_id, score in engine.search(query, n=n):
            if match_id not in matches:
                matches[match_id] = set()
            matches[match_id].add((key, score))

    return matches

def user_search_dart():
    corpus = DartPapers.get_paragraph_corpus()
    engine = BertSentenceSearch(corpus, save_name=DartPapers.__name__, batch_size=256, blacklist=blacklist_doc())
    paragraph_terms = get_paragraph_terms(corpus, engine, n=100)

    while True:
        print("-----------------------------------------------------")
        query = input('>>> ')

        matches = engine.search(query, n=10)
        # print the results of the search
        print('Top 10 matches:')
        for match_id, score in matches:
            raw_text = corpus[match_id]   # get the matching text for the given id
            matching_terms = paragraph_terms.get(match_id, set())
            # print the text
            print(score) 
            print(raw_text)
            for term, score in matching_terms:
                print(f'- {term} ({score})')
            print('')



def main():
    """For each term in the ontology, find matching paragraphs from the DART papers"""

    corpus = DartPapers.get_paragraph_corpus()
    ontology = FlatOntology.get_corpus()


    engine = BertSentenceSearch(corpus, save_name=DartPapers.__name__, batch_size=256, blacklist=blacklist_doc())
    
    # query = ontology['food']

    
    # create a pandas dataframe from the results
    columns=['node', 'query', 'text_id', 'text_chunk', 'text', 'score']
    rows = []
    # for key, query in tqdm([*ontology.items()], desc='performing searches'):
    for key, query in tqdm([*valid_ontology()], desc='performing searches'):
        matches = engine.search(query, n=10)
        for match_id, score in matches:
            raw_text = corpus[match_id]
            text_id, text_chunk = match_id
            rows.append([key, query, text_id, text_chunk, raw_text, score])
    
    #sort rows by score
    rows = sorted(rows, key=lambda x: x[-1], reverse=True)
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('output/uaz_document_concept_matches.csv', index=False)
    


def main2():
    """Find paragraphs in the DART papers that have 2+ matching concepts. First stab at finding causal relationships"""

    corpus = DartPapers.get_paragraph_corpus()
    engine = BertSentenceSearch(corpus, save_name=DartPapers.__name__, batch_size=256, blacklist=blacklist_doc())
    ontology = FlatOntology.get_corpus()

    # create a list of linked concepts
    links = {} # map<paragraph_id, list<(concept_id, score)>>
    embeddings = {}

    #save the embeddings for each concept
    for key, query in tqdm([*valid_ontology()], desc='matching concept pairs'):
        embeddings[key] = engine.embed_query(query).cpu()
        matches = engine.search(query, n=10)
        for match_id, score in matches:
            if match_id not in links:
                links[match_id] = []
            links[match_id].append((key, score))

    #rank all of the concept pairs
    ranked_links = [] #list<tuple<paragraph_id, concept_id, concept_id, score>> where score is the distance between the two concepts + 
    for paragraph_id, concepts in tqdm([*links.items()], desc='ranking concept pairs'):
        if len(concepts) > 1:
            done = set()
            for concept_id1, score1 in concepts:
                for concept_id2, score2 in concepts:
                    if concept_id1 == concept_id2:
                        continue
                    if (concept_id1, concept_id2) in done or (concept_id2, concept_id1) in done:
                        continue
                    done.add((concept_id1, concept_id2))
                    embedding1 = embeddings[concept_id1]
                    embedding2 = embeddings[concept_id2]
                    dist = torch.cosine_similarity(embedding1, embedding2, dim=0)
                    score = score1 * score2 / dist
                    # score_v2 = score1 + score2 - dist
                    ranked_links.append((paragraph_id, concept_id1, concept_id2, score))

    ranked_links = sorted(ranked_links, key=lambda x: x[3])

    # create a pandas dataframe from the results
    columns=['node1', 'node2', 'query1', 'query2', 'paper_id', 'chunk', 'text', 'score']
    rows = []
    for paragraph_id, concept_id1, concept_id2, score in ranked_links:
        raw_text = corpus[paragraph_id]
        text_id, text_chunk = paragraph_id
        rows.append([concept_id1, concept_id2, ontology[concept_id1], ontology[concept_id2], text_id, text_chunk, raw_text, float(score)])

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('output/uaz_document_concept_pairings.csv', index=False)
    exit(1)
    
    
    
    # print the results of the matches
    for paragraph_id, concept_id1, concept_id2, score in ranked_links[:100]:
        print('-------------------------')
        print(f'{corpus[paragraph_id]}')
        print(f'- {concept_id1}: {ontology[concept_id1]}')
        print(f'- {concept_id2}: {ontology[concept_id2]}')
        print(f'- score: {score}')

        print('\n\n\n')


    # # print the results of the matches
    # for paragraph_id, concept_ids in links.items():
    #     if len(concept_ids) > 1:
    #         # print(paragraph_id, concept_ids)
    #         print('-------------------------')
    #         print(f'{corpus[paragraph_id]}')
    #         for concept_id in concept_ids:
    #             print(f'- {concept_id}: {ontology[concept_id]}')

    #         print('\n\n\n')
    
    pdb.set_trace()


def main3():
    uaz_pairings = get_uaz_concept_pairs()
    our_pairings = get_our_concept_pairs()
    corpus = DartPapers.get_paragraph_corpus()

    #construct sets of valid ontology terms, and valid papers
    valid_concepts = set(k for k,_ in valid_ontology())
    valid_papers = set(k for k,_ in corpus.keys())
    

    # create a uaz_map from concept pairs to papers, filtering for valid concepts and papers
    uaz_map = {}
    for concept1, concept2, paper_ids in uaz_pairings:
        if concept1 not in valid_concepts or concept2 not in valid_concepts:
            continue
        if concept1 < concept2:
            concept2, concept1 = concept1, concept2 
        for paper_id in paper_ids:
            if paper_id not in valid_papers:
                continue
            if (concept1, concept2) not in uaz_map:
                uaz_map[(concept1, concept2)] = set()
            uaz_map[(concept1, concept2)].add(paper_id)
           
            
    our_map = {}
    for concept1, concept2, paper_ids in our_pairings:
        if concept1 not in valid_concepts or concept2 not in valid_concepts:
            continue
        if concept1 < concept2:
            concept2, concept1 = concept1, concept2
        for paper_id in paper_ids:
            if paper_id not in valid_papers:
                continue
            if (concept1, concept2) not in our_map:
                our_map[(concept1, concept2)] = set()
            our_map[(concept1, concept2)].add(paper_id)

    
    columns = ['concept1', 'concept2', 'uaz_count', 'our_count', 'num_matches']
    rows = []

    all_pairs = set([*uaz_map.keys()]).union(set([*our_map.keys()]))

    for concept1, concept2 in all_pairs:
        uaz_paper_ids = uaz_map.get((concept1, concept2), set())
        our_paper_ids = our_map.get((concept1, concept2), set())
        uaz_count = len(uaz_paper_ids)
        our_count = len(our_paper_ids)
        num_matches = len(uaz_paper_ids.intersection(our_paper_ids))
        rows.append([concept1, concept2, uaz_count, our_count, num_matches])

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('output/uaz_document_concept_pairings_vs_jataware.csv', index=False)

    exit(1)



def main4():

    uaz_concepts = get_uaz_concepts_to_docs()
    our_concepts = get_our_concepts_to_docs()
    pdb.set_trace()

    pass



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


def get_our_concepts_to_docs():
    #read them in as a dataframe
    df = pd.read_csv('output/uaz_documents_concept_matches.csv')
    # columns are node,query,text_id,text_chunk,text,score
    #keep "node", "text_id"
  
    pdb.set_trace()

def get_uaz_concept_pairs():
    with open('data/statements_2022_march_v4.jsonl') as f:
        lines = f.readlines()
    
    # concepts = {} # concept -> set<doc-id>
    # docs = set() # list of all docs encountered

    pairings = [] # subj:str, obj:str, doc_ids: set<str>

    for line in lines:
        data = json.loads(line)
        subjs = get_concepts(data['subj'])
        objs = get_concepts(data['obj'])
        doc_ids = set(get_docs(data['evidence']))
        if len(doc_ids) == 0:
            continue
        for subj in subjs:
            for obj in objs:
                pairings.append((subj, obj, doc_ids))


    return pairings



def get_our_concept_pairs():
    #read them in as a dataframe from the results of 'output/uaz_document_concept_pairings.csv'
    df = pd.read_csv('output/uaz_document_concept_pairings.csv')

    #keep only the relevant columns from ['node1', 'node2', 'query1', 'query2', 'paper_id', 'chunk', 'text', 'score']
    df = df[['node1', 'node2', 'paper_id']]

    pairings = [] # subj:str, obj:str, doc_id: str

    for _, row in df.iterrows():
        pairings.append((row['node1'], row['node2'], row['paper_id']))

    #squash the pairings by doc_id, so we have pairings: subj:str, obj:str, doc_ids: set<str>
    pair_map = {}
    for subj, obj, doc_id in pairings:
        key = (subj, obj)
        if key not in pair_map:
            pair_map[key] = set()
        pair_map[key].add(doc_id)

    pairings = [(k[0], k[1], v) for k, v in pair_map.items()]

    return pairings
        





    
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
    user_search_dart()
    # main()
    main2()
    # main3()
    # main4()