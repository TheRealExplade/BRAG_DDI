from pyvis.network import Network
from rag.graph import build_graph
from rag.graph import build_graph, get_subgraph

G = build_graph()
G = get_subgraph(G, "warfarin", "aspirin")
G = build_graph()

net = Network(notebook=False, cdn_resources='in_line')

for node in G.nodes:
    net.add_node(node)

for u, v, data in G.edges(data=True):
    net.add_edge(u, v, label=data["relation"])

# 🔥 Generate HTML safely
html = net.generate_html()

# ✅ Write with UTF-8 (fixes your error)
with open("graph.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Graph saved as graph.html")