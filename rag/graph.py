import json
import networkx as nx

def build_graph():
    with open("data/drugbank.json") as f:
        data = json.load(f)

    G = nx.Graph()

    for entry in data:
        d1 = entry["drug1"]
        d2 = entry["drug2"]

        # direct interaction
        G.add_edge(d1, d2, relation="interacts_with")

        # enzymes
        for enzyme in entry.get("enzymes", []):
            G.add_edge(d1, enzyme, relation="metabolized_by")

        # targets
        for target in entry.get("targets", []):
            G.add_edge(d1, target, relation="affects")
            G.add_edge(d2, target, relation="affects")

        # effects
        for effect in entry.get("effects", []):
            G.add_edge(d1, effect, relation="causes")
            G.add_edge(d2, effect, relation="causes")

    return G



def query_graph(G, drug1, drug2):
    valid_paths = []

    for path in nx.all_simple_paths(G, source=drug1, target=drug2, cutoff=3):
        # keep only meaningful biological paths
        if any(node in ["bleeding", "platelets", "clotting_factors"] for node in path):
            valid_paths.append(" → ".join(path))

    return "\n".join(valid_paths) if valid_paths else "No meaningful path found"



def get_subgraph(G, drug1, drug2):
    nodes = set([drug1, drug2])

    for n in G.neighbors(drug1):
        nodes.add(n)

    for n in G.neighbors(drug2):
        nodes.add(n)

    return G.subgraph(nodes)