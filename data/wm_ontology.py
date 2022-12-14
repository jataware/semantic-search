import yaml
from dataclasses import dataclass
from .corpora import Corpus, CorpusLoader
from scipy.sparse import csr_matrix

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
        nodes, _ = FlatOntology.process_ontology()
        return nodes
    
    @staticmethod
    def get_graph() -> dict[str, list[str]]:
        _, graph = FlatOntology.process_ontology()
        return graph

    @staticmethod
    def process_ontology():
        #extract the nodes from the ontology
        with open('data/CompositionalOntology_metadata.yml', 'r') as f:
            data = yaml.safe_load(f)[0]
            assert isinstance(data, dict), 'data is not a dictionary'
        nodes: list[FlatOntology.Node] = []
        graph: dict[str, list[str]] = {} #for constructing csr adjacency matrix
        FlatOntology.extract_nodes(data, nodes, graph)
        
        return nodes, graph

    @staticmethod
    def get_adjacency_matrix() -> tuple[csr_matrix, list[str]]:
        graph = FlatOntology.get_graph()
        # nodes = list(graph.keys())
        node_indices = {node: i for i, node in enumerate(graph.keys())}

        n = len(node_indices)
        data = []
        col = []
        row = []
        for node, children in graph.items():
            # add an edge from the node to itself
            data.append(1)
            col.append(node_indices[node])
            row.append(node_indices[node])
            for child in children:
                # add an edge from the node to its child
                data.append(1)
                col.append(node_indices[child])
                row.append(node_indices[node])

                # add a reciprocal edge from the child to the node
                data.append(1)
                col.append(node_indices[node])
                row.append(node_indices[child])

        adj = csr_matrix((data, (row, col)), shape=(n, n))

        # clip the data to max 1
        adj.data = adj.data.clip(0, 1)

        return adj, node_indices

    @staticmethod
    def get_leaf_nodes() -> list[str]:
        graph = FlatOntology.get_graph()
        return [node for node, children in graph.items() if len(children) == 0]





    @staticmethod
    def extract_nodes(data: dict, nodes: list[Node], graph: dict[str, list[str]]):
        """recursive function to extract the nodes from the ontology"""
        assert 'node' in data, 'dictionary does not contain a node at the top level'
        raw_node = data['node']

        #get the node name and any examples if they exist
        name = raw_node['name']
        examples = tuple(raw_node['examples']) if 'examples' in raw_node else ()
        
        #insert the node into the list of nodes
        nodes.append(FlatOntology.Node(name, examples))

        #recurse on the children
        children = raw_node['children'] if 'children' in raw_node else []
        
        #add the node to the graph
        if name not in graph:
            graph[name] = []
        for child in children:
            child_name = child['node']['name']
            graph[name].append(child_name)

        # recursively process the children
        for child in children:
            FlatOntology.extract_nodes(child, nodes, graph)

    @staticmethod
    def node_to_query_string(node: Node):
        terms = list(node.examples)
        name = node.name.replace('_', ' ')
        if name not in terms:
            terms = [name] + terms
        return ', '.join(terms)