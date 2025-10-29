import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json 

# Set UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Load MongoDB config from .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_Agent")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

def get_content_from_field(doc, field_name):
    """
    Helper to safely extract content from a MongoDB field, handling string, 
    object, list of strings, or list of JSON objects (for user_knowledgebase).
    """
    content = doc.get(field_name)
    
    if content is None:
        return None
        
    if isinstance(content, str):
        return content
    
    # Handle object with 'content' field (like old sections)
    elif isinstance(content, dict) and content.get("content"):
        return content["content"]
        
    # Handle list of items (for guidance or knowledgebase arrays)
    elif isinstance(content, list):
        items = []
        for item in content:
            if isinstance(item, str):
                # For string arrays (like policy guidance)
                items.append(item)
            elif isinstance(item, dict):
                # For JSON/object arrays (like user_knowledgebase)
                try:
                    # Format JSON objects nicely for the prompt
                    items.append(json.dumps(item, indent=2, ensure_ascii=False))
                except Exception:
                    # Fallback for non-JSON-serializable dicts
                    items.append(str(item))
        
        # Use a list format for items that are not raw JSON blocks
        if all(isinstance(item, str) and not item.startswith('{') for item in items):
            return "\n".join([f"- {item}" for item in items])
        else:
            return "\n---\n".join(items)
        
    return None


def build_prompt(agent_name, title, target_chunk, previous_chunk="", next_chunk="", db_name=None, collection_name=None):
    """
    Fetches and returns the seven required fields from the MongoDB document:
    system_prompt, primary_objective, knowledge_base, user_policy_guidance,
    user_knowledgebase, automatic_policy_actions, and do_not_flag.
    Then appends a hardcoded section for Evidence & Mapping Requirements.
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

    # 1. System Prompt
    system_prompt = get_content_from_field(doc, "system_prompt")
    parts.append(system_prompt if system_prompt else "## SYSTEM PROMPT MISSING\n")

    # 2. Primary Objective
    primary_objective = get_content_from_field(doc, "primary_objective")
    parts.append("\n## Primary Objective\n" + (primary_objective if primary_objective else "❌ PRIMARY OBJECTIVE MISSING\n"))
    
    # 3. Knowledge Base
    knowledge_base = get_content_from_field(doc, "knowledge_base")
    parts.append("\n## Knowledge Base\n-----------------\n" + (knowledge_base if knowledge_base else "❌ KNOWLEDGE BASE MISSING\n"))

    # 4. User Knowledgebase
    user_knowledgebase = get_content_from_field(doc, "user_knowledgebase")
    if user_knowledgebase:
        parts.append("\n## User Knowledgebase (JSON Context)\n-----------------------------------\n" + user_knowledgebase)

    # 5. User Policy Guidance
    user_policy_guidance = get_content_from_field(doc, "user_policy_guidance")
    if user_policy_guidance:
        parts.append("\n## User Policy Guidance (Specific Rules)\n---------------------------------------\n" + user_policy_guidance)

    # 6. Automatic Policy Actions
    automatic_policy_actions = get_content_from_field(doc, "automatic_policy_actions")
    if automatic_policy_actions:
        parts.append("\n## Automatic Policy Actions (Hard Rules)\n---------------------------------------\n" + automatic_policy_actions)

    # 7. Do Not Flag / Exemption List
    do_not_flag = get_content_from_field(doc, "do_not_flag")
    if do_not_flag:
        parts.append("\n## Do Not Flag / Exemption List\n------------------------------\n" + do_not_flag)

    # --- Hardcoded Section ---
    evidence_mapping_block = """
---

###  **Evidence & Mapping Requirements**

If flagged, quote **minimal text spans (≤ 50 words)** from **Target Chunk only**.
Each must include a **recommendation** and **confidence** score.

---

###  **Output Format**

```json
{
  "issues_found": "true|false|human",
  "observation": "≤2 sentences (≤40 tokens). Summarize division type or province issue.",
  "spans": [
    {
      "quote": "exact problematic text (≤50 words)",
      "recommendation": "delete|rephrase|fact-check|provide-references",
      "confidence": 0.25|0.5|0.75|1.0
    }
  ]
}
Use an empty spans array if "issues_found" = "false" or "human".
No commentary outside JSON.

###  **Context Rule**

Focus only on the **Target Chunk**; use *preceding* and *next* chunks solely to resolve ambiguous references (e.g., “they,” “the province,” etc.).
Never infer intent beyond textual evidence.

---

"""
    parts.append(evidence_mapping_block)

    inputs_section = f"""## Inputs

* Book Title: {title}
* **Previous_chunk (context only; do not quote if absent in Target Chunk):** {previous_chunk}
* **Target_chunk (review focus):** {target_chunk}
* **Next_chunk (context only; do not quote if absent in Target Chunk):** {next_chunk}

Return **only** the JSON above — no commentary.
"""
    parts.append(inputs_section)

    return "\n\n".join(parts)

if __name__ == "__main__":
    # Example usage
    prompt = build_prompt(
        agent_name="Federal Unity",
        title="Sample Document Title",
        target_chunk="This is the target chunk of text to be evaluated.",
        previous_chunk="This is the previous chunk for context.",
        next_chunk="This is the next chunk for context."
    )
    print(prompt)