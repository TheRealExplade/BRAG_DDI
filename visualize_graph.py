"""Render the mechanistic neighbourhood of a drug pair to an HTML graph.

    python visualize_graph.py                        # warfarin + aspirin
    python visualize_graph.py ketoconazole simvastatin
    python visualize_graph.py warfarin aspirin --shared-drugs 15
"""

import argparse

from pyvis.network import Network

from rag.graph import load_graph, get_subgraph

NODE_COLORS = {
    "drug": "red",
    "target": "green",
    "enzyme": "orange",
    "transporter": "purple",
    "pathway": "blue",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drug1", nargs="?", default="warfarin")
    parser.add_argument("drug2", nargs="?", default="aspirin")
    parser.add_argument(
        "--shared-drugs",
        type=int,
        default=0,
        help="also show up to N drugs that interact with BOTH inputs (default 0)"
    )
    parser.add_argument("--output", default="graph.html")
    args = parser.parse_args()

    G, lookup = load_graph()

    subgraph = get_subgraph(
        G,
        lookup,
        args.drug1,
        args.drug2,
        max_shared_drugs=args.shared_drugs
    )

    if subgraph is None:
        raise SystemExit(
            f"Could not resolve '{args.drug1}' and/or '{args.drug2}' in the DrugBank graph."
        )

    print(
        f"Rendering {subgraph.number_of_nodes()} nodes / "
        f"{subgraph.number_of_edges()} edges"
    )

    net = Network(
        height="900px",
        width="100%",
        notebook=False,
        cdn_resources="in_line"
    )

    for node, attrs in subgraph.nodes(data=True):

        node_type = attrs.get("node_type", "unknown")

        label = attrs.get("name", node) if node_type == "drug" else node

        title = f"Node: {label}\nType: {node_type}"

        if attrs.get("profile_incomplete"):
            title += "\n(no mechanistic profile in drugs.json)"

        net.add_node(
            node,
            label=label,
            title=title,
            color=NODE_COLORS.get(node_type, "gray")
        )

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

    # net.write_html() opens the file with the platform default encoding,
    # which crashes on Windows (cp1252) because DrugBank names and interaction
    # descriptions contain non-ASCII characters. Write UTF-8 explicitly.
    html = net.generate_html(notebook=False)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
