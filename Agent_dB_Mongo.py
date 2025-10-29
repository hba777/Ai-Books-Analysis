import re
import json
from pymongo import MongoClient
import pprint
from dotenv import load_dotenv
import os

# ------------------- Load Environment -------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_Agent")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

# ------------------- All Reviews Data -------------------
all_reviews = {
    "National Security": {
        "criteria": """ - Portrays military operations, strategies, or decisions in a negative light
 - Contradicts official narratives about wars (1965, 1971, etc.)
 - Reveals sensitive information about military or security operations
 - Suggests military failures or incompetence
 - Criticizes military leadership's decision-making""",
        "confidence_score": 80
    },
    "Institutional Integrity": {
        "criteria": """ - Undermines the reputation of state institutions (particularly the Army)
 - Suggests corruption, incompetence, or overreach by institutions
 - Portrays military rule as harmful to the country
 - Suggests institutional failures or abuses of power
 - Criticizes military or intelligence agencies' actions or motivations""",
        "confidence_score": 80
    },
    "Historical Narrative Review": {
        "criteria": """ - Contradicts official historical narratives about key events
 - Criticizes founding leaders or their decisions
 - Provides alternative interpretations of partition or creation of Pakistan
 - Presents the 1971 war in a way that differs from official narrative
 - Questions decisions made by historical leadership""",
        "confidence_score": 80
    },
    "Foreign Relations Review": {
        "criteria": """ - Contains criticism of allied nations (China, Saudi Arabia, Turkey, etc.)
 - Discusses sensitive topics related to allied nations
 - Makes comparisons that could offend foreign partners
 - Suggests policies or actions that contradict official foreign policy
 - Contains language that could harm bilateral relations""",
        "confidence_score": 80
    },
    "Federal Unity Review": {
        "criteria": """ - Creates or reinforces divisions between provinces or ethnic groups
 - Suggests preferential treatment of certain regions or ethnicities
 - Highlights historical grievances between regions
 - Portrays certain ethnic groups as dominating others
 - Discusses separatist movements or provincial alienation""",
        "confidence_score": 80
    },
    "Rhetoric & Tone Review": {
        "criteria": """ - Uses emotionally charged or inflammatory language
 - Contains sweeping generalizations or absolute statements
 - Uses rhetoric that could be divisive or provocative
 - Employs exaggeration or hyperbole on sensitive topics
 - Attributes motives without evidence""",
        "confidence_score": 80
    }
}

# ------------------- Agent Extraction from File -------------------
def extract_structured_content(text):
    structured_data = []
    agent_pattern = re.compile(
        r"#######################################################(.*?)Agent\s*([\s\S]*?)(?=(?:#######################################################|$))",
        re.DOTALL
    )
    matches = agent_pattern.findall(text)
    skip_headings = {
        "## Inputs",
        "## Output JSON Schema",
        "### Human Review",
        "## Decision Algorithm (apply strictly)"
    }

    for agent_title, agent_content in matches:
        agent_name = re.sub(r"^#+\s*", "", agent_title).strip()  # store as agent_name
        sections = []
        lines = agent_content.splitlines()
        current_heading = None
        current_content = []

        for line in lines:
            if line.startswith("#"):
                if current_heading or current_content:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join(current_content).strip()
                    })
                current_heading = line.strip()
                current_content = []
            else:
                current_content.append(line)

        if current_heading or current_content:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(current_content).strip()
            })

        sections = [s for s in sections if s.get("heading") not in skip_headings]
        structured_data.append({"agent_name": agent_name, "sections": sections})

    return structured_data

# ------------------- User Input -------------------
def get_policy_guidance_input():
    agent_name = input("Enter Agent Name: ")
    main_category = input("Enter Main Category: ")
    sub_category = input("Enter Sub Category: ")
    topic = input("Enter Topic: ")

    print("\nEnter Policy Guidance (one per line). Type END on a new line when finished:")
    user_policy_guidance = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        if line.strip():
            user_policy_guidance.append(line.strip())

    print("\nPaste JSON data for this topic. Type END on a new line when finished:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    json_data_str = "\n".join(lines)
    try:
        user_knowledgebase = json.loads(json_data_str)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON! Error: {e}")
        return None

    return {
        "agent_name": agent_name,
        "main_category": main_category,
        "sub_category": sub_category,
        "topic": topic,
        "user_policy_guidance": user_policy_guidance,
        "user_knowledgebase": user_knowledgebase
    }

# ------------------- MongoDB Operations -------------------
def upsert_agent_document(agent_doc):
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    update_doc = {
        "$setOnInsert": {
            "sections": agent_doc.get("sections", []),
            "user_policy_guidance": [],
            "user_knowledgebase": [],
            "type": "analysis"
        }
    }

    collection.update_one({"agent_name": agent_doc["agent_name"]}, update_doc, upsert=True)
    client.close()

def merge_reviews_to_agents(reviews_data):
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    for agent_name, content in reviews_data.items():
        update_doc = {
            "$set": {
                "criteria": content["criteria"],
                "confidence_score": content.get("confidence_score"),
                "type": "analysis"
            },
            "$setOnInsert": {
                "user_policy_guidance": [],
                "user_knowledgebase": []
            }
        }
        collection.update_one({"agent_name": agent_name}, update_doc, upsert=True)

    client.close()

def display_agents():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    print(f"\n--- Agents in Collection '{MONGO_COLLECTION_NAME}' ---")
    for doc in collection.find({}):
        print(f"Agent Name: {doc.get('agent_name')}")
        print(f"Type: {doc.get('type')}")
        print(f"Criteria:\n{doc.get('criteria', '')}")
        print(f"Confidence Score: {doc.get('confidence_score', '')}")
        print(f"Sections: {len(doc.get('sections', []))} sections")
        print(f"User Policy Guidance: {doc.get('user_policy_guidance', [])}")
        print(f"User Knowledgebase: {doc.get('user_knowledgebase', [])}")
        print("-"*60)
    client.close()

# ------------------- Main Execution -------------------
if __name__ == "__main__":
    file_path = "Prompts Revised with KB - 28 Aug 25"

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found.")
        text = None

    if text:
        extracted_agents = extract_structured_content(text)
        for agent in extracted_agents:
            upsert_agent_document(agent)
            print(f"✅ Agent processed: {agent['agent_name']}")

        merge_reviews_to_agents(all_reviews)
        print("✅ All reviews merged into agents.")

        agent_choice = input("\nDo you want to add User Policy Guidance & Knowledgebase? (yes/no): ").strip().lower()
        agent_name = input("Enter Agent Name to display/add data: ").strip()

        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        base_agent = collection.find_one({"agent_name": agent_name})
        if not base_agent:
            print(f"❌ No agent found with name: {agent_name}")
        else:
            if agent_choice == "yes":
                doc = get_policy_guidance_input()
                if doc:
                    update_fields = {}
                    if doc["user_policy_guidance"]:
                        update_fields.setdefault("$push", {}).setdefault(
                            "user_policy_guidance", {"$each": doc["user_policy_guidance"]}
                        )
                    if doc["user_knowledgebase"]:
                        update_fields.setdefault("$push", {}).setdefault(
                            "user_knowledgebase", {"$each": [doc["user_knowledgebase"]]}
                        )
                    update_fields.setdefault("$set", {}).update({
                        "main_category": doc["main_category"],
                        "sub_category": doc["sub_category"],
                        "topic": doc["topic"]
                    })
                    collection.update_one({"agent_name": agent_name}, update_fields, upsert=True)
                    print(f"\n✅ Updated Agent: {doc['agent_name']}")
            else:
                print(f"\n📄 Existing data for Agent: {agent_name}")
                pprint.pprint(base_agent)

        client.close()
        display_agents()
