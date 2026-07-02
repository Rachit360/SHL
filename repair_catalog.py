from pathlib import Path
import re

input_file = Path("data/shl_product_catalog.json")
output_file = Path("data/shl_product_catalog_fixed.json")

text = input_file.read_text(encoding="utf-8")

# Fix Microsoft 365 broken newline
text = re.sub(
    r'"name":\s*"Microsoft\s*\n\s*365 \(New\)"',
    '"name": "Microsoft 365 (New)"',
    text,
)

output_file.write_text(text, encoding="utf-8")

print("✅ Fixed catalog saved as:")
print(output_file)