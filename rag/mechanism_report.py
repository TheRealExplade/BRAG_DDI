# rag/mechanism_report.py
#
# Deterministic, non-LLM DrugBank lookup report for a drug pair (graph RAG,
# not vector RAG -- see note in README / conversation for why).
#
# Produces a structured report that explicitly separates two very different
# kinds of evidence:
#   - documented_interactions: an interaction DrugBank records directly
#     between the two drugs. High-confidence, ground truth.
#   - mechanistic_overlaps: biological entities (targets/enzymes/pathways)
#     the two drugs happen to share. This is a much weaker, rule-based
#     signal and NOT a documented interaction -- it's flagged as such so
#     the LLM (and the pharmacist reading the report) doesn't conflate it
#     with a real DrugBank-confirmed interaction.
#
# Note on data completeness: rag/processed/drugs.json stores targets/
# enzymes/transporters/pathways as plain identifiers with no per-entity
# action (substrate/inhibitor/inducer) and no "carriers" category. Those
# fields are simply omitted here rather than fabricated -- if a richer
# DrugBank export becomes available, extend _drug_profile() below.

import networkx as nx

from rag.graph import load_drugs, resolve_drug_id, get_overlay_meta
from rag.severity import lookup_severity, get_severity_meta

_drugs_by_id = None


def _load_drugs_by_id():
    global _drugs_by_id

    if _drugs_by_id is None:
        _drugs_by_id = {d["drug_id"]: d for d in load_drugs()}

    return _drugs_by_id


def _drug_profile(drug_id, G=None):
    drug = _load_drugs_by_id().get(drug_id)

    if not drug:
        # Known to the interaction graph but absent from drugs.json, so we
        # have no mechanistic profile. Say so instead of omitting it, or the
        # report would silently imply the drug has no targets/enzymes.
        if G is not None and drug_id in G:
            return {
                "name": G.nodes[drug_id].get("name", drug_id),
                "profile_incomplete": True,
                "targets": [],
                "enzymes": [],
                "transporters": [],
                "pathways": [],
            }
        return None

    profile = {
        "name": drug["name"],
        "targets": drug.get("targets", []),
        "enzymes": drug.get("enzymes", []),
        "transporters": drug.get("transporters", []),
        "pathways": drug.get("pathways", []),
    }

    # ATC classification (RxClass) -- drug CLASS, not mechanism. Context for
    # the LLM/pharmacist, not currently used in the shared-entity reasoning
    # below (that's still purely enzyme/target/transporter/pathway based).
    if drug.get("atc_classes"):
        profile["atc_classes"] = drug["atc_classes"]

    # Carry provenance through so the report can disclose which fields came
    # from the hand-curated overlay rather than the DrugBank export.
    if drug.get("_overlay_fields"):
        profile["_overlay_fields"] = drug["_overlay_fields"]

    return profile


def build_mechanism_report(G, lookup, drug1_name, drug2_name):

    names = {"drug1": drug1_name, "drug2": drug2_name}
    ids = {}
    unresolved = []

    for key, name in names.items():
        drug_id = resolve_drug_id(lookup, name)
        if drug_id is None:
            unresolved.append(name)
        ids[key] = drug_id

    report = {
        "source": "DrugBank (deterministic lookup, no ML/LLM)",
        "input_drugs": [
            {"drugbank_id": ids[k], "name": names[k]}
            for k in ("drug1", "drug2") if ids[k] is not None
        ],
        "unresolved_inputs": unresolved,
        "per_drug_profile": {},
        "documented_interactions": [],
        "mechanistic_overlaps": [],
        "reference_severity": None,
    }

    if unresolved:
        return report

    drug1_id, drug2_id = ids["drug1"], ids["drug2"]

    for drug_id in (drug1_id, drug2_id):
        profile = _drug_profile(drug_id, G)
        if profile:
            report["per_drug_profile"][drug_id] = profile

    canonical_a = report["per_drug_profile"].get(drug1_id, {}).get("name")
    canonical_b = report["per_drug_profile"].get(drug2_id, {}).get("name")
    ddinter_match = lookup_severity(drug1_name, drug2_name, canonical_a, canonical_b)

    if ddinter_match:
        report["reference_severity"] = {
            "level": ddinter_match["level"],
            "source": get_severity_meta().get("source", "DDInter"),
            "note": (
                "Curator-assigned severity from an independent database, "
                "NOT this project's own DrugBank interaction data and NOT "
                "an LLM judgment. Treat as a second opinion, not ground truth "
                "-- DDInter's severity criteria may differ from this "
                "project's clinical context."
            ),
        }

    if drug1_id not in G or drug2_id not in G:
        return report

    if G.has_edge(drug1_id, drug2_id):
        edge = G[drug1_id][drug2_id]
        if "interaction" in edge:
            report["documented_interactions"].append({
                "drug_a": drug1_id,
                "drug_b": drug2_id,
                "description": edge["interaction"],
            })

    shared_targets, shared_enzymes = [], []
    shared_transporters, shared_pathways = [], []

    name_a = G.nodes[drug1_id].get("name", drug1_id)
    name_b = G.nodes[drug2_id].get("name", drug2_id)

    for node in nx.common_neighbors(G, drug1_id, drug2_id):

        # Classify by EDGE relation, not node_type. A protein like CYP2C9 is
        # a metabolising enzyme for one drug and a pharmacodynamic target for
        # another; the node carries a single node_type (last writer wins), so
        # only the edge records the role in THIS drug's context.
        edge_a = G[drug1_id][node]
        edge_b = G[drug2_id][node]
        relations = {edge_a.get("relation"), edge_b.get("relation")}

        if "metabolized_by" in relations:
            bucket = shared_enzymes
        elif "transported_by" in relations:
            bucket = shared_transporters
        elif relations == {"pathway"}:
            shared_pathways.append(node)
            continue
        elif "targets" in relations:
            shared_targets.append(node)
            continue
        else:
            continue

        entry = {
            "name": node,
            "actions_a": edge_a.get("actions", []),
            "actions_b": edge_b.get("actions", []),
        }
        entry["likely_pk_effects"] = _infer_pk_effects(
            node, entry["actions_a"], entry["actions_b"], name_a, name_b
        )
        bucket.append(entry)

    if shared_targets or shared_enzymes or shared_transporters or shared_pathways:
        overlap = {
            "drug_a": drug1_id,
            "drug_b": drug2_id,
            "shared_targets": shared_targets,
            "shared_enzymes": shared_enzymes,
            "shared_transporters": shared_transporters,
            "shared_pathways": shared_pathways,
            "note": (
                "Shared biological entities inferred from the graph, "
                "NOT a DrugBank-documented interaction. Treat as a "
                "lower-confidence, rule-based hypothesis."
            ),
        }

        overlay_meta = get_overlay_meta()
        if overlay_meta and _uses_overlay(report["per_drug_profile"]):
            overlap["data_provenance"] = overlay_meta.get("source", "overlay")

        report["mechanistic_overlaps"].append(overlap)

    return report


def _uses_overlay(profiles):
    return any(p.get("_overlay_fields") for p in profiles.values())


def _infer_pk_effects(entity, actions_a, actions_b, name_a, name_b):
    """Rule-based pharmacokinetic reading of a shared enzyme/transporter.

    Deterministic, no LLM. Directionality matters clinically: a substrate
    paired with an inhibitor is a very different risk from two substrates.
    """

    a, b = set(actions_a or []), set(actions_b or [])

    if not a or not b:
        return []

    effects = []

    for sub, other, sub_name, other_name in (
        (a, b, name_a, name_b),
        (b, a, name_b, name_a),
    ):
        if "substrate" not in sub:
            continue

        if "inhibitor" in other:
            effects.append(
                f"{other_name} inhibits {entity}; may REDUCE clearance of "
                f"{sub_name}, increasing its exposure and toxicity risk"
            )

        if "inducer" in other:
            effects.append(
                f"{other_name} induces {entity}; may INCREASE clearance of "
                f"{sub_name}, reducing its efficacy"
            )

    if "substrate" in a and "substrate" in b:
        effects.append(
            f"{name_a} and {name_b} are both {entity} substrates; "
            f"possible competitive metabolism"
        )

    return effects


def format_mechanism_report(report):
    """Render the report as plain text for the LLM prompt's GRAPH section."""

    if report["unresolved_inputs"]:
        return "Could not resolve in DrugBank graph: " + ", ".join(report["unresolved_inputs"])

    lines = []

    atc_lines = []
    for drug_id, profile in report["per_drug_profile"].items():
        if profile.get("atc_classes"):
            names = ", ".join(f"{c['name']} ({c['code']})" for c in profile["atc_classes"][:3])
            atc_lines.append(f"- {profile['name']}: {names}")
    if atc_lines:
        lines.append("DRUG CLASSES (ATC, context only -- not interaction evidence):")
        lines.extend(atc_lines)
        lines.append("")

    if report.get("reference_severity"):
        ref = report["reference_severity"]
        lines.append(
            f"REFERENCE SEVERITY ({ref['source']}): {ref['level']} "
            f"(independent second opinion, not ground truth -- see note in JSON)"
        )
        lines.append("")

    if report["documented_interactions"]:
        lines.append("DOCUMENTED DRUGBANK INTERACTIONS:")
        for item in report["documented_interactions"]:
            lines.append(f"- {item['description']}")
    else:
        lines.append("DOCUMENTED DRUGBANK INTERACTIONS: none found")

    if report["mechanistic_overlaps"]:
        lines.append("")
        lines.append("INFERRED MECHANISTIC OVERLAPS (not documented -- shared biology only):")
        for overlap in report["mechanistic_overlaps"]:
            if overlap["shared_targets"]:
                lines.append("- Shared targets: " + ", ".join(overlap["shared_targets"]))

            for label, key in (("enzyme", "shared_enzymes"),
                               ("transporter", "shared_transporters")):
                for entry in overlap.get(key, []):
                    lines.append(
                        f"- Shared {label} {entry['name']}: "
                        f"{report['input_drugs'][0]['name']}={entry['actions_a'] or ['unknown']}, "
                        f"{report['input_drugs'][1]['name']}={entry['actions_b'] or ['unknown']}"
                    )
                    for effect in entry.get("likely_pk_effects", []):
                        lines.append(f"    -> rule-based PK reading: {effect}")

            if overlap["shared_pathways"]:
                lines.append("- Shared pathways: " + ", ".join(overlap["shared_pathways"]))

            if overlap.get("data_provenance"):
                lines.append(f"  [enzyme/transporter data source: {overlap['data_provenance']}]")

    return "\n".join(lines)
