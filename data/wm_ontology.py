import yaml
from dataclasses import dataclass
from .corpora import Corpus, CorpusLoader


class FlatOntology(CorpusLoader):
    @dataclass(order=True, frozen=True)
    class Node:
        name: str
        examples: tuple[str, ...]

    @staticmethod
    def get_corpus() -> Corpus[str]:
        nodes = FlatOntology.get_nodes()
        docs = {node.name: FlatOntology.node_to_query_string(node) for node in nodes}
        return Corpus(docs)
        
    @staticmethod
    def get_nodes() -> list[Node]:
        #extract the nodes from the ontology
        with open('data/CompositionalOntology_metadata.yml', 'r') as f:
            data = yaml.safe_load(f)[0]
            assert isinstance(data, dict), 'data is not a dictionary'
        nodes: list[FlatOntology.Node] = []
        FlatOntology.extract_nodes(data, nodes)
        
        return nodes


    @staticmethod
    def extract_nodes(data: dict, nodes: list[Node]):
        assert 'node' in data, 'dictionary does not contain a node at the top level'
        raw_node = data['node']

        #get the node name and any examples if they exist
        name = raw_node['name']
        examples = tuple(raw_node['examples']) if 'examples' in raw_node else ()
        
        #insert the node into the list of nodes
        nodes.append(FlatOntology.Node(name, examples))

        #recurse on the children
        children = raw_node['children'] if 'children' in raw_node else []
        
        for child in children:
            FlatOntology.extract_nodes(child, nodes)

    @staticmethod
    def node_to_query_string(node: Node):
        terms = list(node.examples)
        name = node.name.replace('_', ' ')
        if name not in terms:
            terms = [name] + terms
        return ', '.join(terms)
        # return ', '.join([' '.join(node.name.split('_'))] + list(node.examples))
