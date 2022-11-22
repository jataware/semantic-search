import json
import yaml
from dataclasses import dataclass
from tf_idf_search import PlaintextSearch, SklearnSearch
from bert_search import BertSearch
from tqdm import tqdm
import pickle

import pdb


@dataclass(order=True, frozen=True)
class Node:
    name: str
    examples: tuple


def main():
    #extract the nodes from the ontology
    with open('data/wm_flat_metadata.yml', 'r') as f:
        data = yaml.safe_load(f)[0]
        assert isinstance(data, dict), 'data is not a dictionary'
    nodes: list[Node] = []
    extract_nodes(data, nodes)

    #read descriptions from json array
    with open('data/descriptions.json') as f:
        descriptions = json.load(f)

    #create search objects
    engines = {
        'text': PlaintextSearch(descriptions),
        'sklearn': SklearnSearch(descriptions),
        'bert': BertSearch(descriptions)
    }

    matches = {}
    for node in tqdm(nodes, desc='searching for nodes'):
        # name = node.name
        query = node_to_query_string(node)

        matches[node] = {}
        for engine_name, engine in engines.items():
            results = engine.search(query, n=3)
            matches[node][engine_name] = results

    #print out the matches
    for node, match in matches.items():
        print(box_string(f'NODE: "{node.name}"'))
        print(box_string(f'QUERY: "{node_to_query_string(node)}"', sym='·'))
        
        for engine, results in match.items():
            print(f'------------------------------------ [{engine} matches] ---------------------------')
            for text, score in results:
                print(f'(score={score:.2f})\n{text}\n')
        print('\n')

    #save the matches as a pickle
    with open('matches.pkl', 'wb') as f:
        pickle.dump(matches, f)
    
    #save the matches to a json file. First need to convert the nodes to strings
    matches = {str(node): match for node, match in matches.items()}
    with open('matches.json', 'w') as f:
        json.dump(matches, f, indent=4)




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


# def box_string(concept: str, sym:str='#'):
#     s = sym * (len(concept) + 4)
#     s += f'\n{sym} {concept} {sym}\n'
#     s += sym * (len(concept) + 4)
#     return s 

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

