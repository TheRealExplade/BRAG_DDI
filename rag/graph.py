import json
import os
import networkx as nx

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
DRUGS_PATH = os.path.join(PROCESSED_DIR, "drugs.json")
INTERACTIONS_PATH = os.path.join(PROCESSED_DIR, "interactions.json")

# Enzyme/transporter data lives in a separate, version-controlled overlay
# because drugs.json ships with 0% coverage for both and cannot be
# regenerated (no generator script in this repo). See the file's _meta.
OVERLAY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "enzyme_transporter_overlay.json"
)

# DrugBank uses formal/generic names; map common brand/colloquial names
# that show up in demos and user input to the name actually in drugs.json.
# Also doubles as a USAN<->INN crosswalk for matching external datasets
# (e.g. DDInter labels rifampin "Rifampicin", the INN/British name) --
# see rag/severity.py.
COMMON_ALIASES = {
    "aspirin": "acetylsalicylic acid",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",  # INN/British name -- USAN is "acetaminophen"
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "coumadin": "warfarin",
    "rifampicin": "rifampin",
}


def reverse_aliases():
    """canonical name -> set of known alternate names that resolve to it."""
    reverse = {}
    for alias, canonical in COMMON_ALIASES.items():
        reverse.setdefault(canonical, set()).add(alias)
    return reverse


_drugs_cache = None
_overlay_meta = None


def load_overlay():
    """Enzyme/transporter overlay keyed by drug_id, or {} if absent."""
    global _overlay_meta

    if not os.path.exists(OVERLAY_PATH):
        _overlay_meta = None
        return {}

    with open(OVERLAY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    _overlay_meta = data.get("_meta")

    return data.get("drugs", {})


def get_overlay_meta():
    """Provenance of the overlay, so reports can disclose where data came from."""
    if _overlay_meta is None:
        load_overlay()
    return _overlay_meta


def load_drugs():
    """Parsed drugs.json with the enzyme/transporter overlay merged in, cached.

    drugs.json is ~13 MB; it used to be parsed three times per startup
    (build_graph, load_graph, and mechanism_report). Cache it once.

    The overlay only ADDS enzyme/transporter entries -- it never overwrites
    data already present in drugs.json, so a future real DrugBank export
    takes precedence automatically.
    """
    global _drugs_cache

    if _drugs_cache is None:
        with open(DRUGS_PATH, encoding="utf-8") as f:
            drugs = json.load(f)

        overlay = load_overlay()

        if overlay:
            for drug in drugs:
                extra = overlay.get(drug["drug_id"])

                if not extra:
                    continue

                for field in ("enzymes", "transporters"):
                    if not drug.get(field) and extra.get(field):
                        drug[field] = extra[field]
                        drug.setdefault("_overlay_fields", []).append(field)

        _drugs_cache = drugs

    return _drugs_cache


def entity_name(entity):
    """Accept both the flat form ('CYP2C9') and the rich form
    ({'name': 'CYP2C9', 'actions': [...]})."""
    if isinstance(entity, dict):
        return entity.get("name")
    return entity


def entity_actions(entity):
    if isinstance(entity, dict):
        return entity.get("actions", [])
    return []


def build_graph():

    G = nx.Graph()

    drugs = load_drugs()

    with open(INTERACTIONS_PATH, encoding="utf-8") as f:
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

            name = entity_name(target)

            if not name:
                continue

            G.add_node(
                name,
                node_type="target"
            )

            G.add_edge(
                drug_id,
                name,
                relation="targets",
                actions=entity_actions(target)
            )

        # Enzymes
        #
        # 'actions' (substrate / inhibitor / inducer) describe the drug's
        # RELATIONSHIP to the enzyme, so they belong on the edge, not the
        # node. This is what lets mechanism_report tell "A is a substrate,
        # B inhibits it" apart from a bare shared-enzyme coincidence.

        for enzyme in drug["enzymes"]:

            name = entity_name(enzyme)

            if not name:
                continue

            G.add_node(
                name,
                node_type="enzyme"
            )

            G.add_edge(
                drug_id,
                name,
                relation="metabolized_by",
                actions=entity_actions(enzyme)
            )

        # Transporters

        for transporter in drug.get("transporters", []):

            name = entity_name(transporter)

            if not name:
                continue

            G.add_node(
                name,
                node_type="transporter"
            )

            G.add_edge(
                drug_id,
                name,
                relation="transported_by",
                actions=entity_actions(transporter)
            )

        # Pathways

        for pathway in drug["pathways"]:

            name = entity_name(pathway)

            if not name:
                continue

            G.add_node(
                name,
                node_type="pathway"
            )

            G.add_edge(
                drug_id,
                name,
                relation="pathway"
            )

    # ------------------
    # Interaction edges
    # ------------------

    # interactions.json references some drug_ids that are absent from
    # drugs.json (e.g. DB09368). add_edge() would silently create bare,
    # attribute-less nodes for those, which then get misread as non-drug
    # "shared entities" by query_graph(). Register them as proper drug
    # nodes -- using the name carried on the interaction record -- so the
    # graph stays internally consistent and the interactions aren't lost.

    for interaction in interactions:

        for id_key, name_key in (("drug1_id", "drug1"), ("drug2_id", "drug2")):

            drug_id = interaction[id_key]

            if drug_id not in G:

                G.add_node(
                    drug_id,
                    node_type="drug",
                    name=interaction.get(name_key, drug_id),
                    profile_incomplete=True
                )

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


def resolve_drug_id(lookup, name):

    key = name.lower().strip()

    if key in lookup:
        return lookup[key]

    alias = COMMON_ALIASES.get(key)

    if alias and alias in lookup:
        return lookup[alias]

    return None


def query_graph(
    G,
    lookup,
    drug1_name,
    drug2_name
):

    drug1_id = resolve_drug_id(lookup, drug1_name)
    drug2_id = resolve_drug_id(lookup, drug2_name)

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

            elif node_type == "transporter":
                context.append(
                    f"Shared transporter: {node}"
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
    drug2_name,
    max_shared_drugs=0
):
    """Mechanistic neighbourhood of a drug pair, safe to render.

    Deliberately EXCLUDES each drug's interacting-drug neighbours by default.
    A well-studied drug interacts with thousands of others, and G.subgraph()
    keeps every edge among the selected nodes -- for warfarin + aspirin that
    is ~2.5k nodes and ~944k edges, which hangs any renderer. The useful
    view is the drugs plus the biological entities they act on.

    max_shared_drugs > 0 additionally includes up to that many drugs that
    interact with BOTH inputs (bounded, so it stays renderable).
    """

    drug1 = resolve_drug_id(lookup, drug1_name)
    drug2 = resolve_drug_id(lookup, drug2_name)

    if not drug1 or not drug2:
        return None

    if drug1 not in G or drug2 not in G:
        return None

    nodes = {drug1, drug2}

    # non-drug neighbours only (targets / enzymes / transporters / pathways)
    for drug in (drug1, drug2):
        for neighbor in G.neighbors(drug):
            if G.nodes[neighbor].get("node_type") != "drug":
                nodes.add(neighbor)

    if max_shared_drugs > 0:
        shared = [
            n for n in nx.common_neighbors(G, drug1, drug2)
            if G.nodes[n].get("node_type") == "drug"
        ]
        nodes.update(shared[:max_shared_drugs])

    return G.subgraph(nodes)


def load_graph():

    drugs = load_drugs()

    lookup = build_name_lookup(drugs)

    G = build_graph()

    return G, lookup