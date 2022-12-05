# run in root folder: `python -m experiments.uaz_comparison`
import json
import yaml
from dataclasses import dataclass
from search.tf_idf_search import PlaintextSearch, SklearnSearch
from search.bert_search import BertSearch
from search.corpora import Corpus
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

import pdb


@dataclass(order=True, frozen=True)
class Node:
    name: str
    examples: tuple

@dataclass
class Indicator:
    name: str
    display_name: str
    description: str
    dataset: str

def get_uaz_results():

    with open('data/indicators.jsonl') as f:
        lines = f.readlines()
        indicators = [json.loads(line) for line in lines]

    corpus_docs = []
    name_to_key = {}
    indicator_map: dict[int, Indicator] = {}
    for indicator in indicators:
        # doc_names.append(indicator['_source']['name'])
        for out in indicator['_source']['outputs']:
            #display name, description, unit, unit description
            key = len(corpus_docs)
            doc = \
f"""name: {out['name']};
display name: {out['display_name']};
description: {out['description']};
unit: {out['unit']};
unit description: {out['unit_description']};"""
            corpus_docs.append(doc)
            name_to_key[out['name']] = key
            indicator_map[key] = Indicator(out['name'], out['display_name'], out['description'], indicator['_source']['name'])
    corpus = Corpus.from_list(corpus_docs)

    with open('data/indicators_with_uaz_matches.jsonl', 'r') as f:
        lines = f.readlines()
        indicators = [json.loads(line) for line in lines]
    
    get_node_name = lambda name: name.split('/')[-1]

    inversion_dict = defaultdict(list) # inversion_dict[node][indicator] = score

    for indicator in tqdm(indicators):
        for output in indicator['outputs']:
            output_name = output['name']
            assert output_name in name_to_key, f'"{output_name}" not in name_to_text'

            ontologies = output['ontologies']
            scores = []
            try:
                for concept in ontologies['concepts']:
                    scores.append((concept['score'], get_node_name(concept['name'])))
            except:
                pass
            try:
                for property in ontologies ['properties']:
                    scores.append((property['score'], get_node_name(property['name'])))
            except:
                pass
            try:
                for process in ontologies['processes']:
                    scores.append((process['score'], get_node_name(process['name'])))
            except:
                pass

            #insert the scores into the inversion dict
            for score, node_name in scores:
                inversion_dict[node_name].append((output_name, score))
    

    #sort the scores in the inversion dict
    for node_name, scores in inversion_dict.items():
        scores.sort(key=lambda x: x[1], reverse=True)
        inversion_dict[node_name] = scores
    
    # #DEBUG print out the top 3 scores for each node
    # for node_name, scores in inversion_dict.items():
    #     print(box_string(f'NODE: "{node_name}"'))
    #     print(box_string(f'QUERY: "{node_name}"', sym='·'))
    #     for indicator_name, score in scores[:3]:
    #         print(f'(score={score:.2f})\n{indicator_name}\n')
    #     print('\n')

    return inversion_dict, corpus, name_to_key, indicator_map


def main():
    #extract the nodes from the ontology
    with open('data/CompositionalOntology_metadata.yml', 'r') as f:
        data = yaml.safe_load(f)[0]
        assert isinstance(data, dict), 'data is not a dictionary'
    nodes: list[Node] = []
    extract_nodes(data, nodes)


    inversion_dict, corpus, name_to_key, indicator_map = get_uaz_results()

    #create search objects
    engines = {
        # 'tf-idf': PlaintextSearch(corpus),
        # 'sklearn': SklearnSearch(descriptions),
        'bert': BertSearch(corpus)
    }
    main_engine = 'bert'

    
    matches = {}
    for node in tqdm(nodes, desc='searching for nodes'):
        # name = node.name
        query = node_to_query_string(node)

        matches[node] = {}
        for engine_name, engine in engines.items():
            results = engine.search(query, n=3)
            matches[node][engine_name] = results

    #save results into dataframe
    rows = [] # matcher,query node,query string,dataset,indicator,display name,description,score
    for node, match in matches.items():
        for engine, results in match.items():
            for key, score in results:
                indicator = indicator_map[key]
                rows.append([engine, node.name, node_to_query_string(node), indicator.dataset, indicator.name, indicator.display_name, indicator.description, score])
        for name, score in inversion_dict[node.name][:3]:
            key = name_to_key[name]
            indicator = indicator_map[key]
            assert indicator.name == name, f'{indicator.name} != {name}'
            rows.append(['UAZ', node.name, node_to_query_string(node), indicator.dataset, indicator.name, indicator.display_name, indicator.description, score])

    df = pd.DataFrame(rows, columns=['matcher', 'query node', 'query string', 'dataset', 'indicator', 'display name', 'description', 'score'])
    df.to_csv('output/ranked_concepts.csv', index=False)

    # save the alternate format comparison
    convert_to_jata_vs_uaz(df, 'output/jata_vs_uaz_v2.csv', engine=main_engine)


    #count agreement/disagreement between bert and uaz
    all_empty = 0
    all_agree = 0
    all_disagree = 0
    some_agree = 0
    for node, match in matches.items():
        results = match[main_engine]
        engine_results = set([indicator_map[result[0]].name for result in results])
        uaz_results = set([indicator_map[name_to_key[name]].name for name, score in inversion_dict[node.name][:3]])

        if engine_results == uaz_results:
            if len(engine_results) == 0:
                all_empty += 1
            else:
                all_agree += 1
        elif engine_results.isdisjoint(uaz_results):
            all_disagree += 1
        else:
            some_agree += 1

    print(f'all agree: {all_agree}')
    print(f'all empty: {all_empty}')
    print(f'all disagree: {all_disagree}')
    print(f'some agree: {some_agree}')


    pass


    # #print out the matches
    # for node, match in matches.items():
    #     print(box_string(f'NODE: "{node.name}"'))
    #     print(box_string(f'QUERY: "{node_to_query_string(node)}"', sym='·'))
        
    #     for engine, results in match.items():
    #         print(f'------------------------------------ [{engine} matches] ---------------------------')
    #         for text, score in results:
    #             print(f'(score={score:.2f})\n{text_to_name[text]}\n{text}\n')

    #     print(f'------------------------------------ [UAZ matches] ---------------------------')
    #     for name, score in inversion_dict[node.name][:3]:
    #         print(f'(score={score:.2f})\n{name}\n{name_to_text[name]}\n')
        
    #     print('\n')

    # #save the matches as a pickle
    # with open('matches.pkl', 'wb') as f:
    #     pickle.dump(matches, f)
    
    # #save the matches to a json file. First need to convert the nodes to strings
    # matches = {str(node): match for node, match in matches.items()}
    # with open('matches.json', 'w') as f:
    #     json.dump(matches, f, indent=4)




def extract_nodes(data: dict, nodes: list[Node]):
    assert 'node' in data, 'dictionary does not contain a node at the top level'
    raw_node = data['node']

    #get the node name and any examples if they exist
    name = raw_node['name']
    examples = tuple(raw_node['examples']) if 'examples' in raw_node else ()
    
    #insert the node into the list of nodes
    nodes.append(Node(name, examples))

    #recurse on the children
    children = raw_node['children'] if 'children' in raw_node else []
    
    for child in children:
        extract_nodes(child, nodes)

    
def node_to_query_string(node: Node):
    return ', '.join([' '.join(node.name.split('_'))] + list(node.examples))


def box_string(concept: str, sym:str='#'):
    lines = concept.split('\n')
    max_len = max([len(line) for line in lines])
    s = sym * (max_len + 4)
    for line in lines:
        s += f'\n{sym} {line:<{max_len}} {sym}'
    s += f'\n{sym * (max_len + 4)}'
    return s


def convert_to_jata_vs_uaz(df: pd.DataFrame, path: str, engine: str='bert'):
    #TODO: make this use the indicators dataclass
    
    uaz_q = set(df[df['matcher']=='UAZ']['query node'].unique())
    bert_q = set(df[df['matcher']==engine]['query node'].unique())
    print(f"{len(uaz_q)} concepts matched by UAZ")
    print(f"{len(bert_q)} concepts matched by Semantic Search")
    
    matched = uaz_q & bert_q
    print(f"{len(matched)} concepts were matched by both systems")

    matches = {}
    results = []
    for q in matched:
        bert_best = df[(df['query node']==q) & (df['matcher']==engine)].sort_values(by='score', ascending=False).iloc[0]
        uaz_best = df[(df['query node']==q) & (df['matcher']=='UAZ')].sort_values(by='score', ascending=False).iloc[0]
        matches[q] = {engine: bert_best, 'uaz': uaz_best}
        
        jata = f"""Dataset: {bert_best.dataset}\n
Indicator: {bert_best.indicator}\n
Display Name: {bert_best['display name']}\n
Description: {bert_best['description']}"""
        
        uaz = f"""Dataset: {uaz_best.dataset}\n
Indicator: {uaz_best.indicator}\n
Display Name: {uaz_best['display name']}\n
Description: {uaz_best['description']}""" 
        
        res = dict(concept=bert_best['query node'],
                query=bert_best['query string'],
                jataware=jata,
                uaz=uaz)
        results.append(res)

    results_df = pd.DataFrame(results)

    results_df.to_csv(path, index=False)






def indicator_ontologies(data, ontologies):
    """
    A function to map UAZ ontologies back into
    the submitted "partial" indicator object
    Params:
        - data: the indicator object
        - ontologies: object from UAZ endpoint
    """
    ontology_dict = {"outputs": {}, "qualifier_outputs": {}}

    # Reorganize UAZ response
    for ontology in ontologies["outputs"]:
        ontology_dict["outputs"][ontology["name"]] = ontology["ontologies"]

    for ontology in ontologies["qualifier_outputs"]:
        ontology_dict["qualifier_outputs"][ontology["name"]] = ontology["ontologies"]

    # Map back into partial indicator object to build complete indicator
    for output in data["outputs"]:
        output["ontologies"] = ontology_dict["outputs"][output["name"]]

    if data.get("qualifier_outputs", None):
        for qualifier_output in data["qualifier_outputs"]:
            qualifier_output["ontologies"] = ontology_dict["qualifier_outputs"][qualifier_output["name"]]

    return data

def fetch_ontologies_for_wdi():
    import gzip as gz
    import requests
    import time

    inds = []
    with open('../data/indicators_with_uaz_matches.jsonl','r') as f:
        for line in f:
            inds.append(json.loads(line.split('\n')[0]))
    
    url = "http://linking.cs.arizona.edu/v1"
    uaz_threshold = 0.6
    uaz_hits = 10
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    params = f"?maxHits={uaz_hits}&threshold={uaz_threshold}&compositional=true"
    type_ = "groundIndicator"
    url_ = f"{url}/{type_}{params}"

    reprocessed = []
    for ind in inds:
        response = requests.put(url_, json=ind, headers=headers)
        ind_new = indicator_ontologies(ind, response.json())
        reprocessed.append(ind_new)
        time.sleep(1)

    with gz.open('../output/indicators_v2.jsonl.gz','a') as f:
        for i in reprocessed:
            out = json.dumps(i)+'\n'
            f.write(out.encode('utf-8'))




if __name__ == '__main__':
    main()

