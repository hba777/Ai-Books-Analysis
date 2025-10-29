import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List, Dict, Any

# ------------------- Load Environment -------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_Agent")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

# --- Review Data ---
# This dictionary maps file names (after cleaning) to the corresponding review criteria key.
FILENAME_TO_REVIEW_MAP = {
    "National Security": "National Security",
    "Instituitional Integrity": "Institutional Integrity", # Handles user's file name spelling
    "Historical": "Historical Narrative Review",
    "Foreign Policy": "Foreign Relations Review",
    "Federal Unity": "Federal Unity Review",
    "Rhetoric": "Rhetoric & Tone Review"
}

# The source of the criteria and confidence scores provided by the user.
ALL_REVIEWS = {
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

# --- MongoDB Connection Setup ---
client = None
db = None
collection = None

try:
    if not MONGO_URI or not MONGO_DB_NAME or not MONGO_COLLECTION_NAME:
         raise ValueError("One or more MongoDB environment variables are missing.")
        
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    print(f"✅ Connected to MongoDB: Database='{MONGO_DB_NAME}', Collection='{MONGO_COLLECTION_NAME}'")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    collection = None 

# ------------------------------------------------------------------------------------------------------

def extract_and_store_agent_data(file_paths: List[str]):
    """
    Extracts System Prompt and all H2 headings with their content blocks,
    and stores the data in MongoDB based on the specific field requirements, 
    including criteria and confidence score from ALL_REVIEWS.
    """
    
    if collection is None:
        print("🛑 Cannot process files: MongoDB connection failed.")
        return

    # REGEX setup remains the same for accurate extraction
    h2_heading_pattern = r'^(##[ \t\xa0]*)(?!\#)(.*)$'
    delimiter_pattern = re.compile(r'^-+\s*$', re.MULTILINE)
    
    SPECIFIC_HEADINGS = [
        "Primary Objective", 
        "Knowledge Base", 
        "Automatic Policy Actions", 
        "Do NOT Flag"
    ]

    for file_path in file_paths:
        agent_document: Dict[str, Any] = {}
        
        # --- 1. Extract Agent Name and Map to Review Criteria ---
        base_name = os.path.basename(file_path)
        # agent_name will be, e.g., "National Security", "Instituitional Integrity"
        file_agent_name = os.path.splitext(base_name)[0].replace('_', ' ')
        agent_document["agent_name"] = file_agent_name
        
        # Look up the appropriate key from ALL_REVIEWS using the map
        review_key = FILENAME_TO_REVIEW_MAP.get(file_agent_name)
        
        # --- 2. Fetch Criteria and Confidence Score ---
        if review_key and review_key in ALL_REVIEWS:
            review_data = ALL_REVIEWS[review_key]
            
            # ADDING CRITERIA FIELD
            agent_document["criteria"] = review_data["criteria"].strip()
            
            # SETTING CONFIDENCE SCORE FROM DICTIONARY
            agent_document["confidence_score"] = review_data["confidence_score"]
        else:
            print(f"⚠️ Warning: Review data not found for agent: {file_agent_name}. Using default values.")
            agent_document["criteria"] = "Criteria not available."
            agent_document["confidence_score"] = 0 # Default low score if data is missing
            
        # --- Add Empty Array Fields (Reserved) ---
        agent_document["user_knowledgebase"] = []
        agent_document["user_policy_guidence"] = []
        agent_document["type"] = "analysis" # Fixed field
        
        try:
            if not os.path.exists(file_path):
                print(f"\n--- ⚠️ File not found: {file_path} ---")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            delimiter_match = delimiter_pattern.search(content)

            if delimiter_match:
                # --- 3. Extract System Prompt ---
                system_prompt_text = content[:delimiter_match.start()].strip()
                main_content_body = content[delimiter_match.end():].strip()
                
                agent_document["system_prompt"] = system_prompt_text
                
                # --- 4. Extract Headings and their Content Blocks ---
                
                matches = list(re.finditer(h2_heading_pattern, main_content_body, re.MULTILINE))
                other_headings: Dict[str, str] = {}
                
                for i, match in enumerate(matches):
                    heading_text_full = match.group(2).strip() 
                    heading_text_clean = heading_text_full.replace('**', '').strip()
                    
                    start = match.end()
                    end = matches[i+1].start() if i + 1 < len(matches) else len(main_content_body)
                    
                    content_under_heading = main_content_body[start:end].strip()
                    
                    if heading_text_clean in SPECIFIC_HEADINGS:
                        key = heading_text_clean.lower().replace(' ', '_').strip()
                        key = re.sub(r'[^a-z0-9_]', '', key) 
                        agent_document[key] = content_under_heading
                    else:
                        other_headings[heading_text_full] = content_under_heading

                agent_document["other_headings"] = other_headings
                
                # --- 5. Store in MongoDB ---
                result = collection.insert_one(agent_document)
                print(f"✅ Stored document for '{file_agent_name}'. ID: {result.inserted_id}")
                
            else:
                print(f"❌ Delimiter '---' not found in {file_path}. Skipping.")
                

        except Exception as e:
            print(f"\n--- ❌ Error processing file {file_path}: {e} ---")

# ------------------- Example Usage -------------------
# Define the list of file paths. 
file_list = [
    "prompt1/Federal_Unity.txt",
    "prompt1/Instituitional_Integrity.txt",
    "prompt1/National_Security.txt",
    "prompt1/Rhetoric.txt",
    "prompt1/Historical.txt",
    "prompt1/Foreign_Policy.txt"
]

# Run the function
extract_and_store_agent_data(file_list)