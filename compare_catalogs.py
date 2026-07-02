import json

with open("data/shl_product_catalog_fixed.json", encoding="utf-8") as f:
    data = json.load(f)

matches = []

for item in data:
    name = item["name"].lower()

    if "verify" in name:
        matches.append(item)

print(f"\nFound {len(matches)} Verify products\n")

for item in matches:
    print("=" * 80)
    print("Name :", item["name"])
    print("Keys :", item.get("keys"))
    print("Duration :", item.get("duration"))
    print("Job Levels :", item.get("job_levels"))
    print("Link :", item["link"])