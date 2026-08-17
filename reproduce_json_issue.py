import json
import re
from typing import Any

def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Initial JSONDecodeError: {e}")
        # Try to find the JSON block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e2:
                print(f"Second JSONDecodeError: {e2}")
                # Try to fix unescaped quotes in common places
                # This is a bit hacky but can help
                # Replace "notes": "..." with something safer if there are nested quotes
                # Actually, let's try a simpler fix: 
                # If it's "Expecting ',' delimiter", it might be an unescaped quote.
                pass
        raise

# Example of failing JSON
failing_text = """
{
  "ratings": [
    {
      "trait_index": 376,
      "notes": "He is a "hero" of the story"
    }
  ]
}
"""

try:
    print(extract_json(failing_text))
except Exception as e:
    print(f"Failed as expected: {e}")
