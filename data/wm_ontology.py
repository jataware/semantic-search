import yaml
from dataclasses import dataclass
from .corpora import Corpus, CorpusLoader

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




class WorldModelersOntology(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus[str]:
        #extract the nodes from the ontology
        with open('data/CompositionalOntology_metadata.yml', 'r') as f:
            data = yaml.safe_load(f)[0]
            assert isinstance(data, dict), 'data is not a dictionary'
        nodes: list[Node] = []
        extract_nodes(data, nodes)




        pdb.set_trace()
