import json
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Set UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Load MongoDB config from .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_Agent")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

def build_prompt(agent_name, title, target_chunk, previous_chunk="", next_chunk="", db_name=None, collection_name=None):
    """
    Builds a structured prompt for a given agent by fetching content from MongoDB.
    Compatible with updated logic where user_policy_guidance is a list of strings
    and user_knowledgebase is a list of JSON dicts (multiple updates allowed).
    """
    db_name = db_name or MONGO_DB_NAME
    collection_name = collection_name or MONGO_COLLECTION_NAME

    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ismaster')
    except Exception as e:
        return f"❌ Error connecting to MongoDB: {e}"

    db = client[db_name]
    collection = db[collection_name]

    # Fetch single agent document
    doc = collection.find_one({"agent_name": agent_name})
    if not doc:
        return f"❌ No agent found with name: {agent_name}"

    parts = []

    # Helper to add section by matching heading
    def add_section(title, match_text):
        for s in doc.get("sections", []):
            heading = (s.get("heading") or "").lower()
            if match_text.lower() in heading:
                parts.append(f"## {title}\n" + "-"*len(title) + "\n" + s.get("content", ""))
                return

    # 1. Introduction
    intro_section = None
    for s in doc.get("sections", []):
        heading = s.get("heading")
        if heading and "intro" in heading.lower():
            intro_section = s
            break
    if not intro_section:
        for s in doc.get("sections", []):
            if not s.get("heading"):
                intro_section = s
                break
    if intro_section:
        parts.append("\n" + intro_section.get("content", ""))

    # 2. Primary Objective
    add_section("Primary Objective", "primary objective")

    # 2a. Hardcoded Inputs
    inputs_section = f"""## Inputs

* Book Title: {title}
* **Previous_chunk (for reference only):** {previous_chunk}
* **Target_chunk (review focus):** {target_chunk}
* **Next_chunk (for reference only):** {next_chunk}
"""
    parts.append(inputs_section)

    # 3. Knowledge Base
    kb_parts = []

    # Official Narrative
    for s in doc.get("sections", []):
        heading = (s.get("heading") or "").lower()
        if "official narrative" in heading:
            kb_parts.append(f"{s.get('heading')}\n{s.get('content')}")

    # Key Points
    key_points = []
    for s in doc.get("sections", []):
        heading = (s.get("heading") or "").lower()
        if "key points" in heading:
            for line in s.get("content").splitlines():
                cleaned = line.lstrip("0123456789. ").strip()
                if cleaned:
                    key_points.append(cleaned)

    # Merge User Policy Guidance
    if isinstance(doc.get("user_policy_guidance"), list):
        key_points.extend(doc["user_policy_guidance"])

    if key_points:
        kb_parts.append("### Key Points\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(key_points)]))

    # Sensitive Aspects
    for s in doc.get("sections", []):
        heading = (s.get("heading") or "").lower()
        if "sensitive" in heading:
            kb_parts.append(f"{s.get('heading')}\n{s.get('content')}")

    # Recommended Terminology
    for s in doc.get("sections", []):
        heading = (s.get("heading") or "").lower()
        if "terminology" in heading:
            kb_parts.append(f"{s.get('heading')}\n{s.get('content')}")

    # Authoritative Sources
    for s in doc.get("sections", []):
        heading = (s.get("heading") or "").lower()
        if "sources" in heading:
            kb_parts.append(f"{s.get('heading')}\n{s.get('content')}")

    if kb_parts:
        parts.append("## Knowledge Base\n-----------------\n" + "\n\n".join(kb_parts))

    # 4. Hard Scope Filter
    #add_section("Hard Scope Filter", "hard scope filter")

    # 5. Decision Framework
    add_section("Decision Framework", "Decision Framework")

    # 6. User Knowledgebase JSON
    if isinstance(doc.get("user_knowledgebase"), list) and doc["user_knowledgebase"]:
        kb_json = ""
        for idx, kb in enumerate(doc["user_knowledgebase"], 1):
            try:
                kb_json += f"\n### KB Entry {idx}\n" + json.dumps(kb, indent=2, ensure_ascii=False) + "\n"
            except Exception:
                continue
        if kb_json:
            parts.append("## Some other Knowledge Base to follow:\n--------------------------\n" + kb_json)

    # 7. Human Review + Output Schema
    human_review_block = """## Evidence & Mapping Requirements

If you flag:

spans: list one or more exact minimal quotes (≤50 words each) from the Target Chunk only.

For each span, return:

quote (verbatim text),

suggested_rewrite (neutral, policy-aligned rephrase that preserves meaning if possible; else advise deletion).

If multiple issues exist, include multiple spans. Keep quotes minimal (no more than needed to evidence the claim).

If you do **not** flag:

* Provide a brief observation stating **why it fails the Pakistan Link Test** or **Evidence Test**.

---
### Human Review

* If ambiguous phrasing or insufficient evidence, return `"chunk_flagged": "human"`.

---

## Output JSON Schema
**Crucial:** Return **ONLY** the JSON object, do not add any extra text, comments, or conversational sentences before or after the JSON.

Return **only** this JSON object, Please **strictly follow JSON Formate**, Must follow it please :

```json
{
  "chunk_flagged": "true|false|human",
  "observation": "Brief reasoning (≤120 words). If flagged, include rule references.",
  "spans": [
    {
      "quote": "exact text from target chunk only"
    }
  ],
  "recommendation": "delete|rephrase|fact-check|provide references",
  "confidence": 0.0
}
- All keys (`chunk_flagged`, `observation`, `spans`, `recommendation`, `confidence`) must always be present in the returned JSON object.
- When `chunk_flagged` is "true", the `spans` array MUST contain one or more JSON objects, each with a "quote" key and the exact flagged text from the `target_chunk`.
Use an empty array for spans when chunk_flagged is "false" or "human".
"""
####  CHANGE IN PROMPT
# If ambiguous or insufficient evidence, set "chunk_flagged": "human" and explain ambiguity in observation.

## Decision Algorithm (apply strictly) 

# 1. **Scan for Pakistan linkage** (explicit mentions or unambiguous references). If none → `"false"`.
# 2. If linked, **extract minimal offending span(s)**. If none → `"false"`.
# 3. **Test against Decision Rules** (one-sided, delegitimizing, misrepresenting, sensitive detail). If none → `"false"`.
# 4. If present but **ambiguous/uncited**, return `"human"`.
# 5. Otherwise, `"true"`, include spans + mappings.

    parts.append(human_review_block)
    return "\n\n".join(parts)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python generate_prompt.py <agent_name> <book_title> <target_chunk> <previous_chunk> <next_chunk>")
        sys.exit(1)

    agent = sys.argv[1]
    title = sys.argv[2]
    target_chunk = sys.argv[3]
    previous_chunk = sys.argv[4]
    next_chunk = sys.argv[5]

    prompt = build_prompt(agent, title, target_chunk, previous_chunk, next_chunk)
    print(prompt)