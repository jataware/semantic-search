import json
import yaml
from dataclasses import dataclass
from tf_idf_search import PlaintextSearch, SklearnSearch
from bert_search import BertSearch
from tqdm import tqdm


@dataclass(order=True, frozen=True)
class Node:
    name: str
    examples: list


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
        # 'bert': BertSearch(descriptions)
    }

    matches = {}
    for node in tqdm(nodes, desc='searching for nodes'):
        name = node.name
        query = node_to_query_string(node)

        matches[name] = {}
        for engine_name, engine in engines.items():
            results = engine.search(query, n=3)
            matches[name][engine_name] = results

    #print out the matches
    for name, match in matches.items():
        print(concept_to_boxed_string(name))
        for engine, results in match.items():
            print(f'------------------------------------ [{engine} matches] ---------------------------')
            for text, score in results:
                print(f'(score={score:.2f})\n{text}\n')
        print('\n')




def extract_nodes(data: dict, nodes: list[Node]):
    assert 'node' in data, 'dictionary does not contain a node at the top level'
    raw_node = data['node']

    #get the node name and any examples if they exist
    name = raw_node['name']
    try:
        examples = raw_node['examples']
    except KeyError:
        examples = []
    
    #insert the node into the list of nodes
    nodes.append(Node(name, examples))

    #recurse on the children
    try:
        children = raw_node['children']
    except KeyError:
        children = []
    
    for child in children:
        extract_nodes(child, nodes)

    
def node_to_query_string(node: Node):
    return ', '.join([' '.join(node.name.split('_'))] + node.examples)


def concept_to_boxed_string(concept: str):
    s = '#' * (len(concept) + 4)
    s += f'\n# {concept} #\n'
    s += '#' * (len(concept) + 4)
    return s 


if __name__ == '__main__':
    main()

