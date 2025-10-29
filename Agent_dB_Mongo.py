import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List, Dict, Any

# ------------------- Load Environment -------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_Agent")  # Corrected variable name to match .env structure
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

# --- MongoDB Connection Setup ---
try:
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set in environment variables.")
        
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    print(f"✅ Connected to MongoDB: Database='{MONGO_DB_NAME}', Collection='{MONGO_COLLECTION_NAME}'")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    # Use a dummy collection for testing if connection fails, or exit
    # For production code, you would exit here
    client = None
    db = None
    collection = None

def extract_and_store_agent_data(file_paths: List[str]):
    """
    Extracts the System Prompt, H2 headings, and constructs a document 
    for each file, then stores it in MongoDB.

    Args:
        file_paths (list): A list of full paths to the files.
    """
    if not collection:
        print("🛑 Cannot process files: MongoDB collection not initialized.")
        return

    # REGEX for H2 headings: Matches lines starting with '##', followed by optional whitespace,
    # and strictly NOT followed by another '#'.
    h2_heading_pattern = r'^(##[ \t\xa0]*)(?!\#)(.*)$'
    
    # Delimiter to find the end of the initial System Prompt block
    delimiter_pattern = re.compile(r'^-+\s*$', re.MULTILINE)

    # Specific headings to extract as separate fields
    SPECIFIC_HEADINGS = [
        "Primary Objective", 
        "Knowledge Base", 
        "Automatic Policy Actions", 
        "Do NOT Flag"
    ]

    for file_path in file_paths:
        agent_document: Dict[str, Any] = {}
        
        # --- 1. Extract Agent Name from File Path ---
        # e.g., 'prompt1/National_Security.txt' -> 'National_Security'
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
                
                # Find all H2 headings and their content
                # The regex captures the full heading line, including '##' and text.
                # Since we need the content under the heading, we split the content 
                # by the headings.
                
                # Find all heading matches
                matches = list(re.finditer(h2_heading_pattern, main_content_body, re.MULTILINE))
                
                # Prepare dictionaries for storage
                extracted_headings: Dict[str, str] = {}
                
                # Default arrays for lists/policy items
                agent_document["user_knowledgebase"] = []
                agent_document["user_policy_guidence"] = []

                # Iterate through all matches to get heading and its content
                for i, match in enumerate(matches):
                    # The text of the heading (e.g., 'Primary Objective')
                    heading_text = match.group(2).strip() 
                    
                    # The start of the current heading
                    start = match.end()
                    
                    # The end of the content is the start of the next heading, or end of file
                    end = matches[i+1].start() if i + 1 < len(matches) else len(main_content_body)
                    
                    # Content under the heading
                    content_under_heading = main_content_body[start:end].strip()
                    
                    # Clean the content (e.g., remove list markers, empty lines)
                    cleaned_content = content_under_heading.replace('*', '').replace('-', '').strip()
                    
                    # Store based on specific heading names (case-insensitive check)
                    normalized_heading = heading_text.lower().replace('**', '').strip()
                    
                    if "knowledge base" in normalized_heading and cleaned_content:
                        # Split by newline and filter out empty strings for array storage
                        agent_document["user_knowledgebase"] = [line.strip() for line in cleaned_content.split('\n') if line.strip()]
                    elif "policy actions" in normalized_heading and cleaned_content:
                        # Split by newline and filter out empty strings for array storage
                        agent_document["user_policy_guidence"] = [line.strip() for line in cleaned_content.split('\n') if line.strip()]
                    elif heading_text.replace('**','').strip() in SPECIFIC_HEADINGS:
                        # Store specific headings as separate fields
                        key = heading_text.lower().replace(' ', '_').replace('**', '').replace('🎯_', '').replace('📘_', '').replace('🚫_', '').strip()
                        agent_document[key] = cleaned_content
                    else:
                        # Store all other H2 headings in a generic dictionary
                        extracted_headings[heading_text] = cleaned_content

                
                
                # --- 4. Add Fixed/Template Fields ---
                agent_document["type"] = "analysis"
                agent_document["confidence_score"] = 80 # Stored as an integer
                
                # --- 5. Store in MongoDB ---
                if collection:
                    result = collection.insert_one(agent_document)
                    print(f"✅ Stored document for '{agent_name}'. ID: {result.inserted_id}")
                
            else:
                print(f"❌ Delimiter '---' not found in {file_path}. Skipping.")
                

        except Exception as e:
            print(f"\n--- ❌ Error processing file {file_path}: {e} ---")

# --- Example Usage ---

# Define the list of file paths based on your example.
file_list = [
    "prompt1/Federal_Unity.txt",
    "prompt1/Instituitional_Integrity.txt",
    "prompt1/National_Security.txt",
    "prompt1/Rhetoric.txt",
    "prompt1/Historical.txt",
    "prompt1/Foreign_Policy.txt"
]

# Run the function
# NOTE: Ensure the files in file_list exist in the correct location 
# relative to where you run this script.
# For example, create a 'prompt1' folder and put the files inside it.
extract_and_store_agent_data(file_list)