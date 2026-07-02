import json

with open("data/assessments.json", encoding="utf-8") as f:
    ours = json.load(f)

print("Our assessments:", len(ours))

types = {}

for a in ours:
    t = a.get("test_type_label", "Unknown")
    types[t] = types.get(t, 0) + 1

print("\nAssessment Types")
for k, v in types.items():
    print(k, v)