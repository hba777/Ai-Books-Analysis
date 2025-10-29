import pymongo
import os
import json
from dotenv import load_dotenv
import os
load_dotenv()
# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_KB")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_KB")

# --- Configuration: SET YOUR BASE KNOWLEDGE BASE FOLDER PATH HERE ---
BASE_KB_FOLDER_PATH = 'Knowledge_Base_New' # Aapka existing folder path

def create_mongo_connection():
    """
    MongoDB se connection banata hai.
    :return: MongoDB database object if successful, None otherwise.
    """
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        print(f"Successfully connected to MongoDB database: {MONGO_DB_NAME}")
        return client, db # Client bhi return karein taake finally block mein close kar sakein
    except pymongo.errors.ConnectionFailure as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Please ensure MongoDB server is running on localhost:27017.")
        return None, None

def insert_kb_entry_mongo(collection, main_category, sub_category, topic, json_data_str):
    """
    MongoDB collection mein ek naya knowledge base entry insert karta hai.
    Duplicate topics ko handle karta hai (agar topic pehle se exist karta ho to update karega).
    :param collection: MongoDB collection object.
    :param main_category: The main category name.
    :param sub_category: The sub-category name.
    :param topic: The specific topic name.
    :param json_data_str: The actual JSON content for the topic (as a string).
    :return: True if insertion/update was successful, False otherwise.
    """
    document = {
        "main_category": main_category,
        "sub_category": sub_category,
        "topic": topic,
        "json_data": json_data_str # JSON content ko string ke roop mein store karein, jaise SQLite mein tha
    }
    try:
        # topic ko unique banaya hai SQLite mein, toh yahan upsert use kar sakte hain
        # ya replace_one (jaisa aapne agent configs mein kiya tha).
        # Upsert: Agar topic exist karta hai to update karega, nahi to insert karega.
        result = collection.update_one(
            {"topic": topic}, # Filter by topic
            {"$set": document}, # Document ke saare fields ko set karein
            upsert=True # Agar document nahi mila to insert kar de
        )
        if result.upserted_id:
            return True # Successfully inserted
        elif result.modified_count > 0:
            return True # Successfully updated
        else:
            return False # No change, maybe document was identical
    except pymongo.errors.PyMongoError as e:
        print(f"Error inserting/updating entry for topic '{topic}': {e}")
        return False

def populate_data_from_folders_mongo(db_client, collection_name, base_folder_path):
    """
    MongoDB collection ko specified folder structure se JSON files read karke populate karta hai.
    Collection ko insert karne se pehle clear karta hai.
    :param db_client: MongoDB client object.
    :param collection_name: Collection ka naam.
    :param base_folder_path: The root directory containing your main categories.
    """
    db = db_client[MONGO_DB_NAME] # db_client se database access karein
    collection = db[collection_name]

    # Clear existing data in the collection for a fresh start
    print(f"\n--- Clearing existing data in MongoDB collection '{collection_name}' ---")
    delete_result = collection.delete_many({})
    print(f"Deleted {delete_result.deleted_count} existing documents.")

    print(f"\nPopulating data from folder structure: {base_folder_path}...")
    if not os.path.exists(base_folder_path):
        print(f"Error: Base folder path '{base_folder_path}' does not exist. Please check the path.")
        return

    documents_to_insert = [] # Batch insert ke liye list banayenge

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_folder_path):
        relative_path = os.path.relpath(root, base_folder_path)
        path_parts = [p for p in relative_path.split(os.sep) if p]

        main_category = None
        sub_category = None

        if len(path_parts) >= 1:
            main_category = path_parts[0]
        if len(path_parts) >= 2:
            sub_category = path_parts[1]

        for file_name in files:
            if file_name.endswith('.json'):
                topic_name = os.path.splitext(file_name)[0]
                file_path = os.path.join(root, file_name)

                if not main_category:
                    print(f"Warning: Skipping {file_path} as main_category could not be determined.")
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data_content = f.read() # Read the raw JSON string
                        # Optionally, validate JSON here if needed
                        # json.loads(json_data_content)

                    # Har entry ke liye document prepare karein
                    document = {
                        "main_category": main_category,
                        "sub_category": sub_category,
                        "topic": topic_name,
                        "json_data": json_data_content # JSON content ko string ke roop mein store karein
                    }
                    documents_to_insert.append(document)
                    print(f"Prepared: Main='{main_category}', Sub='{sub_category}', Topic='{topic_name}' from {file_name}")

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file '{file_path}': {e}")
                except Exception as e:
                    print(f"An error occurred while processing '{file_path}': {e}")
    
    # Batch insert all documents
    if documents_to_insert:
        try:
            collection.insert_many(documents_to_insert)
            print(f"\nSuccessfully inserted {len(documents_to_insert)} documents into MongoDB.")
        except pymongo.errors.PyMongoError as e:
            print(f"Error during batch insert: {e}")
    else:
        print("\nNo documents to insert.")

    print("Data population from folders complete.")

def get_all_kb_entries_mongo(collection):
    """
    MongoDB collection se saari entries retrieve karta hai.
    :param collection: MongoDB collection object.
    :return: A list of dictionaries, where each dictionary represents a document.
    """
    # Sort order jaisa SQLite mein tha: main_category, sub_category, topic
    return list(collection.find({}).sort([
        ("main_category", pymongo.ASCENDING),
        ("sub_category", pymongo.ASCENDING),
        ("topic", pymongo.ASCENDING)
    ]))

def get_topics_by_main_category_mongo(collection, main_category):
    """
    Specific main category ke saare topics retrieve karta hai.
    :param collection: MongoDB collection object.
    :param main_category: The name of the main category.
    :return: A list of dictionaries.
    """
    return list(collection.find({"main_category": main_category}).sort([
        ("sub_category", pymongo.ASCENDING),
        ("topic", pymongo.ASCENDING)
    ]))

def get_topics_by_subcategory_mongo(collection, main_category, sub_category):
    """
    Specific main category aur subcategory ke saare topics retrieve karta hai.
    :param collection: MongoDB collection object.
    :param main_category: The name of the main category.
    :param sub_category: The name of the subcategory.
    :return: A list of dictionaries.
    """
    return list(collection.find({"main_category": main_category, "sub_category": sub_category}).sort([
        ("topic", pymongo.ASCENDING)
    ]))

def get_json_data_by_topic_mongo(collection, topic_name):
    """
    Specific topic ke liye JSON data string retrieve karta hai.
    :param collection: MongoDB collection object.
    :param topic_name: The name of the topic.
    :return: The JSON data string if found, None otherwise.
    """
    doc = collection.find_one({"topic": topic_name})
    return doc.get("json_data") if doc else None

def main():
    """
    Main function to run the MongoDB knowledge base operations.
    It sets up the connection, populates it with data from JSON files,
    and demonstrates various queries.
    """
    mongo_client, mongo_db = create_mongo_connection()
    # Corrected check: Use 'is not None' instead of direct boolean evaluation
    if mongo_client is not None and mongo_db is not None:
        # Collection access
        kb_collection = mongo_db[MONGO_COLLECTION_NAME]

        # Populate the collection by reading JSON files from your specified folder
        populate_data_from_folders_mongo(mongo_client, MONGO_COLLECTION_NAME, BASE_KB_FOLDER_PATH)

        # --- Demonstration of Query Functions (MongoDB) ---

        print("\n" + "="*50)
        print("--- All KB Entries (Flat Structure) ---")
        print("="*50)
        all_entries = get_all_kb_entries_mongo(kb_collection)
        for entry in all_entries:
            # MongoDB documents are dictionaries, use .get() for safety
            print(f"ID: {entry.get('_id')}, Main: {entry.get('main_category')}, Sub: {entry.get('sub_category')}, Topic: {entry.get('topic')}")
            # print(entry) # Uncomment to see the full document including JSON data

        print("\n" + "="*50)
        print("--- Topics under 'Official_narratives' ---")
        print("="*50)
        official_narrative_topics = get_topics_by_main_category_mongo(kb_collection, 'Official_narratives')
        for topic in official_narrative_topics:
            print(f"ID: {topic.get('_id')}, Main: {topic.get('main_category')}, Sub: {topic.get('sub_category')}, Topic: {topic.get('topic')}")

        print("\n" + "="*50)
        print("--- Topics under 'Official_narratives' -> 'institutions' ---")
        print("="*50)
        institutions_topics = get_topics_by_subcategory_mongo(kb_collection, 'Official_narratives', 'institutions')
        for topic in institutions_topics:
            print(f"ID: {topic.get('_id')}, Main: {topic.get('main_category')}, Sub: {topic.get('sub_category')}, Topic: {topic.get('topic')}")

        print("\n" + "="*50)
        print("--- Example: JSON data for a specific topic (e.g., 'military') ---")
        print("="*50)
        json_content_military = get_json_data_by_topic_mongo(kb_collection, 'military')
        if json_content_military:
            # Parse the JSON string to pretty-print it
            parsed_json = json.loads(json_content_military)
            print(json.dumps(parsed_json, indent=2))
        else:
            print("Topic 'military' not found or has no JSON data.")

        # Close the MongoDB connection
        mongo_client.close()
        print(f"\nMongoDB connection to {MONGO_DB_NAME} closed.")
    else:
        print("Failed to establish MongoDB connection. Exiting.")


if __name__ == '__main__':
    main()
