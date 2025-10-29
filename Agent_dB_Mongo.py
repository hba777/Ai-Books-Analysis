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

# --- MongoDB Connection Setup ---
client = None
db = None
collection = None

try:
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set in environment variables.")
        
    # Check if database/collection names were loaded correctly (preventing initial error)
    if not MONGO_DB_NAME or not MONGO_COLLECTION_NAME:
         raise ValueError("MONGO_DB_NAME or MONGO_COLLECTION_NAME is not set in environment variables.")
        
    client = MongoClient(MONGO_URI)
    # The following lines now execute only if the environment variables are valid strings
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    print(f"✅ Connected to MongoDB: Database='{MONGO_DB_NAME}', Collection='{MONGO_COLLECTION_NAME}'")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    # collection remains None if connection fails
    client = None
    db = None
    collection = None

# ------------------------------------------------------------------------------------------------------

def extract_and_store_agent_data(file_paths: List[str]):
    """
    Extracts the System Prompt, H2 headings, and constructs a document 
    for each file, then stores it in MongoDB.
    """
    
    # -------------------------------------------------------------
    # CORRECTED CHECK: Use 'is None' as required by pymongo
    # -------------------------------------------------------------
    if collection is None:
        print("🛑 Cannot process files: MongoDB collection not initialized (connection failed).")
        return
    # -------------------------------------------------------------

    # REGEX for H2 headings: Matches lines starting with '##', followed by optional whitespace,
    # and strictly NOT followed by another '#'. It captures the full heading text.
    h2_heading_pattern = r'^(##[ \t\xa0]*)(?!\#)(.*)$'
    
    # Delimiter to find the end of the initial System Prompt block
    delimiter_pattern = re.compile(r'^-+\s*$', re.MULTILINE)

    # Specific headings to extract as separate fields in the document
    SPECIFIC_HEADINGS = [
        "Primary Objective", 
        "Knowledge Base", 
        "Automatic Policy Actions", 
        "Do NOT Flag"
    ]

    for file_path in file_paths:
        agent_document: Dict[str, Any] = {}
        
        # --- 1. Extract Agent Name from File Path ---
        # e.g., 'prompt1/National_Security.txt' -> 'National Security'
        base_name = os.path.basename(file_path)
        agent_name = os.path.splitext(base_name)[0].replace('_', ' ')
        agent_document["agent_name"] = agent_name
        
        try:
            if not os.path.exists(file_path):
                print(f"\n--- ⚠️ File not found: {file_path} ---")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            delimiter_match = delimiter_pattern.search(content)

            if delimiter_match:
                # --- 2. Extract System Prompt ---
                system_prompt_text = content[:delimiter_match.start()].strip()
                main_content_body = content[delimiter_match.end():].strip()
                
                agent_document["system_prompt"] = system_prompt_text
                
                # --- 3. Extract and Categorize H2 Headings and Content ---
                
                # Find all H2 headings
                matches = list(re.finditer(h2_heading_pattern, main_content_body, re.MULTILINE))
                
                extracted_headings: Dict[str, str] = {}
                agent_document["user_knowledgebase"] = []
                agent_document["user_policy_guidence"] = []

                # Iterate through all matches to get heading and its content block
                for i, match in enumerate(matches):
                    # match.group(2) is the actual heading text (without '##')
                    heading_text_full = match.group(2).strip() 
                    
                    # The start of the content block is after the current heading
                    start = match.end()
                    
                    # The end of the content block is the start of the next heading, or end of file
                    end = matches[i+1].start() if i + 1 < len(matches) else len(main_content_body)
                    
                    content_under_heading = main_content_body[start:end].strip()
                    
                    # Remove markdown bolding for clean keys/content
                    heading_text_clean = heading_text_full.replace('**', '').strip()
                    cleaned_content = content_under_heading.strip()
                    
                    # Store based on specific heading names (case-insensitive check)
                    normalized_heading = heading_text_clean.lower()
                    
                    if "knowledge base" in normalized_heading and cleaned_content:
                        # Store items as an array, splitting by newline
                        agent_document["user_knowledgebase"] = [line.strip().lstrip('*- ').strip() for line in cleaned_content.split('\n') if line.strip()]
                    elif "policy actions" in normalized_heading and cleaned_content:
                        # Store items as an array, splitting by newline
                        agent_document["user_policy_guidence"] = [line.strip().lstrip('*- ').strip() for line in cleaned_content.split('\n') if line.strip()]
                    elif heading_text_clean in SPECIFIC_HEADINGS:
                        # Store specific headings as separate fields (e.g., primary_objective)
                        key = heading_text_clean.lower().replace(' ', '_').strip()
                        # Clean up emojis/icons from the key name if present (🎯, 📘, 🚫)
                        key = re.sub(r'[^a-z0-9_]', '', key) 
                        agent_document[key] = cleaned_content
                    else:
                        # Store all other H2 headings in a generic dictionary
                        extracted_headings[heading_text_clean] = cleaned_content

                agent_document["other_headings"] = extracted_headings
                
                # --- 4. Add Fixed/Template Fields ---
                agent_document["type"] = "analysis"
                agent_document["confidence_score"] = 80
                
                # --- 5. Store in MongoDB ---
                result = collection.insert_one(agent_document)
                print(f"✅ Stored document for '{agent_name}'. ID: {result.inserted_id}")
                
            else:
                print(f"❌ Delimiter '---' not found in {file_path}. Skipping.")
                

        except Exception as e:
            print(f"\n--- ❌ Error processing file {file_path}: {e} ---")

# ------------------- Example Usage -------------------

# Define the list of file paths. Ensure these paths are correct relative to your script location.
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