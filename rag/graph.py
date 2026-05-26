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

    drug1 = drug1.lower()
    drug2 = drug2.lower()

    results = []

    if drug1 not in G or drug2 not in G:
        return "No graph relationship found"

    # only direct neighbors
    for neighbor in G.neighbors(drug1):

        if neighbor == drug2:
            results.append(f"{drug1} → {drug2}")

        elif G.has_edge(neighbor, drug2):

            relation1 = G[drug1][neighbor].get("relation", "")
            relation2 = G[neighbor][drug2].get("relation", "")

            results.append(
                f"{drug1} --({relation1})-> {neighbor} --({relation2})-> {drug2}"
            )

    return "\n".join(results[:3])



def get_subgraph(G, drug1, drug2):
    nodes = set([drug1, drug2])

    for n in G.neighbors(drug1):
        nodes.add(n)

    for n in G.neighbors(drug2):
        nodes.add(n)

    return G.subgraph(nodes)