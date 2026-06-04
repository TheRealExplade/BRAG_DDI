import json
from graph import load_graph, query_graph

G, lookup = load_graph()

print("warfarin:", lookup.get("warfarin"))
print("aspirin:", lookup.get("aspirin"))

with open("rag/processed/drugs.json") as f:
    drugs = json.load(f)

for d in drugs:
    if d["name"].lower() == "warfarin":
        print(d)

for d in drugs:
    if d["name"].lower() == "aspirin":
        print(d)

