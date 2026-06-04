from pyvis.network import Network
from rag.graph import load_graph
import networkx as nx

# Load graph + lookup
G, lookup = load_graph()

drug1 = "warfarin"
drug2 = "acetylsalicylic acid"

drug1_id = lookup.get(drug1.lower())
drug2_id = lookup.get(drug2.lower())

if not drug1_id or not drug2_id:
    raise ValueError("Drug not found")

# Build local neighborhood
nodes = {drug1_id, drug2_id}

# Drug1 neighbors
for n in G.neighbors(drug1_id):
    nodes.add(n)

# Drug2 neighbors
for n in G.neighbors(drug2_id):
    nodes.add(n)

subgraph = G.subgraph(nodes)

net = Network(
    height="900px",
    width="100%",
    notebook=False,
    cdn_resources="in_line"
)

# Nodes
for node, attrs in subgraph.nodes(data=True):

    node_type = attrs.get("node_type", "unknown")

    label = node

    if node_type == "drug":
        label = attrs.get("name", node)

    title = f"""
    Node: {label}
    Type: {node_type}
    """

    color = "gray"

    if node_type == "drug":
        color = "red"

    elif node_type == "target":
        color = "green"

    elif node_type == "enzyme":
        color = "orange"

    elif node_type == "pathway":
        color = "blue"

    net.add_node(
        node,
        label=label,
        title=title,
        color=color
    )

# Edges
for u, v, attrs in subgraph.edges(data=True):

    relation = attrs.get("relation", "")

    title = relation

    if "interaction" in attrs:
        title += f"\n\n{attrs['interaction']}"

    net.add_edge(
        u,
        v,
        label=relation,
        title=title
    )

net.write_html("graph.html")

print("Saved graph.html")