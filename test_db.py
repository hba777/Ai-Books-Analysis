import pymongo
import json
import os
from dotenv import load_dotenv
import re

# Load environment variables from a .env file
load_dotenv()

# --- MongoDB Configuration (Hardcoded as per request) ---
DB_NAME = "LLM_Responses"
COLLECTION_NAME = "agent_responses"
# Make sure your MONGO_URI is set in your .env file
MONGO_URI = os.getenv("MONGO_URI")

def get_agent_responses_from_mongo(agent_name: str = None):
    """
    Connects to the MongoDB, retrieves documents from the 'agent_responses'
    collection, and prints their content in a structured format with
    each key-value pair on a separate line.
    
    Args:
        agent_name (str, optional): The name of the agent to filter by.
                                     If None, all documents are retrieved.
    """
    mongo_client = None
    try:
        if not MONGO_URI:
            print("❌ Error: MONGO_URI not found in environment variables.")
            return

        mongo_client = pymongo.MongoClient(MONGO_URI)
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]

        query = {}
        if agent_name:
            query["agent_name"] = agent_name
        
        document_count = collection.count_documents(query)
        print(f"✅ Successfully connected to the database and found {document_count} documents.")
        
        documents = collection.find(query)
        
        for doc in documents:
            print("-" * 50)
            print(f"Document ID: {doc.get('_id')}")
            print(f"Agent Name: {doc.get('agent_name')}")
            print(f"Timestamp: {doc.get('timestamp')}")
            
            response_content_str = doc.get("response_content", "{}")
            
            try:
                # Use regex to find and extract the JSON part
                match = re.search(r'\{.*\}', response_content_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    parsed_content = json.loads(json_str)
                    print("\nParsed Response Content:")
                    
                    # Print each key and value on a new line
                    for key, value in parsed_content.items():
                        # Handle spans separately if it's a list
                        if key == "spans" and isinstance(value, list):
                            print(f"  {key}:")
                            for span_item in value:
                                # assuming each item is a dict with 'quote'
                                if isinstance(span_item, dict) and 'quote' in span_item:
                                    print(f"    - Quote: {span_item['quote']}")
                                else:
                                    print(f"    - {span_item}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print("\n❌ JSON not found in response_content.")
                    print(f"Raw Content: {response_content_str}")
            except json.JSONDecodeError as e:
                print(f"\n❌ Error decoding JSON from response_content: {e}")
                print(f"Raw Content: {response_content_str}")

    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if mongo_client:
            mongo_client.close()

if __name__ == "__main__":
    print("--- Fetching ALL documents ---")
    get_agent_responses_from_mongo()
    
    # Example usage for a specific agent (uncomment to use)
    # print("\n--- Fetching documents for 'Foreign Relations Review' agent ---")
    # get_agent_responses_from_mongo(agent_name="Foreign Relations Review")