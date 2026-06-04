import json
import networkx as nx

def build_graph():

    G = nx.Graph()

    with open("rag\processed\drugs.json") as f:
        drugs = json.load(f)


    with open("rag\processed\interactions.json") as f:
        interactions = json.load(f)

    # ------------------
    # Drug nodes
    # ------------------

    for drug in drugs:

        drug_id = drug["drug_id"]

        if drug_id is None:
            print("Missing drug ID:", drug["name"])
            continue

        G.add_node(
            drug_id,
            node_type="drug",
            name=drug["name"],
            indication=drug["indication"],
            mechanism=drug["mechanism"]
        )

        # Targets

        for target in drug["targets"]:

            G.add_node(
                target,
                node_type="target"
            )

            G.add_edge(
                drug_id,
                target,
                relation="targets"
            )

        # Enzymes

        for enzyme in drug["enzymes"]:

            G.add_node(
                enzyme,
                node_type="enzyme"
            )

            G.add_edge(
                drug_id,
                enzyme,
                relation="metabolized_by"
            )

        # Pathways

        for pathway in drug["pathways"]:

            G.add_node(
                pathway,
                node_type="pathway"
            )

            G.add_edge(
                drug_id,
                pathway,
                relation="pathway"
            )

    # ------------------
    # Interaction edges
    # ------------------

    for interaction in interactions:

        G.add_edge(
            interaction["drug1_id"],
            interaction["drug2_id"],

            relation="interacts_with",

            interaction=interaction["interaction"]
        )

    return G


def build_name_lookup(drugs):

    lookup = {}

    for drug in drugs:

        lookup[
            drug["name"].lower()
        ] = drug["drug_id"]

    return lookup


def query_graph(
    G,
    lookup,
    drug1_name,
    drug2_name
):

    drug1_id = lookup.get(drug1_name.lower())
    drug2_id = lookup.get(drug2_name.lower())

    if not drug1_id or not drug2_id:
        return "Drug not found"

    if drug1_id not in G or drug2_id not in G:
        return "No graph relationship found"

    context = []

    # Direct interaction

    if G.has_edge(drug1_id, drug2_id):

        edge = G[drug1_id][drug2_id]

        interaction = edge.get(
            "interaction",
            ""
        )

        context.append(
            f"Direct interaction: {interaction}"
        )

    common = []

    for node in nx.common_neighbors(
        G,
        drug1_id,
        drug2_id
    ):

        node_type = G.nodes[node].get("node_type")

        if node_type != "drug":
            common.append(node)

    if common:

        context.append(
            "Shared graph entities:"
        )

        for node in common[:10]:

            node_type = G.nodes[node].get(
                "node_type",
                "unknown"
            )

            if node_type == "target":
                context.append(
                    f"Shared target: {node}"
                )

            elif node_type == "enzyme":
                context.append(
                    f"Shared enzyme: {node}"
                )

            elif node_type == "pathway":
                context.append(
                    f"Shared pathway: {node}"
                )

    return "\n".join(context)



def get_subgraph(
    G,
    lookup,
    drug1_name,
    drug2_name
):

    drug1 = lookup.get(drug1_name.lower())
    drug2 = lookup.get(drug2_name.lower())

    if not drug1 or not drug2:
        return None


def load_graph():

    with open("rag\processed\drugs.json") as f:
        drugs = json.load(f)

    lookup = build_name_lookup(drugs)

    G = build_graph()

    return G, lookup