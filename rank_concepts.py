import json
import yaml
from dataclasses import dataclass
from tf_idf_search import PlaintextSearch, SklearnSearch
from bert_search import BertSearch
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle

import pdb


@dataclass(order=True, frozen=True)
class Node:
    name: str
    examples: tuple


def get_uaz_results():

    with open('data/indicators.jsonl') as f:
        lines = f.readlines()
        indicators = [json.loads(line) for line in lines]

    corpus = []
    name_to_text = {}
    text_to_name = {}
    text_to_display_name = {}
    text_to_description = {}
    text_to_indicator = {}
    for indicator in indicators:
        # doc_names.append(indicator['_source']['name'])
        for out in indicator['_source']['outputs']:
            #display name, description, unit, unit description
            description = \
f"""name: {out['name']};
display name: {out['display_name']};
description: {out['description']};
unit: {out['unit']};
unit description: {out['unit_description']};"""
            corpus.append(description)
            name_to_text[out['name']] = description
            text_to_name[description] = out['name']
            text_to_display_name[description] = out['display_name']
            text_to_description[description] = out['description']
            text_to_indicator[description] = indicator['_source']['name']


    with open('data/indicators_with_uaz_matches.jsonl', 'r') as f:
        lines = f.readlines()
        indicators = [json.loads(line) for line in lines]
    
    get_node_name = lambda name: name.split('/')[-1]

    inversion_dict = defaultdict(list) # inversion_dict[node][indicator] = score

    for indicator in tqdm(indicators):
        indicator_name = indicator['name']
        for output in indicator['outputs']:
            output_name = output['name']
            assert output_name in name_to_text, f'"{output_name}" not in name_to_text'

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

    return inversion_dict, corpus, name_to_text, text_to_name, text_to_display_name, text_to_description, text_to_indicator


def main():
    #extract the nodes from the ontology
    with open('data/CompositionalOntology_metadata.yml', 'r') as f:
        data = yaml.safe_load(f)[0]
        assert isinstance(data, dict), 'data is not a dictionary'
    nodes: list[Node] = []
    extract_nodes(data, nodes)

    # #read descriptions from json array
    # with open('data/descriptions.json') as f:
    #     descriptions = json.load(f)

    inversion_dict, corpus, name_to_text, text_to_name, text_to_display_name, text_to_description, text_to_indicator = get_uaz_results()

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
            for text, score in results:
                rows.append([engine, node.name, node_to_query_string(node), text_to_name[text], text_to_indicator[text], text_to_display_name[text], text_to_description[text], score])
        for name, score in inversion_dict[node.name][:3]:
            text = name_to_text[name]
            rows.append(['UAZ', node.name, node_to_query_string(node), name, text_to_indicator[text], text_to_display_name[text], text_to_description[text], score])

    df = pd.DataFrame(rows, columns=['matcher', 'query node', 'query string', 'dataset', 'indicator', 'display name', 'description', 'score'])
    df.to_csv('data/ranked_concepts.csv', index=False)
    


    #count agreement/disagreement between bert and uaz
    all_empty = 0
    all_agree = 0
    all_disagree = 0
    some_agree = 0
    for node, match in matches.items():
        results = match[main_engine]
        engine_results = set([text_to_indicator[result[0]] for result in results])
        uaz_results = set([text_to_indicator[name_to_text[name]] for name, score in inversion_dict[node.name][:3]])

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

    pdb.set_trace()

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


if __name__ == '__main__':
    main()

