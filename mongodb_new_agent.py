import pymongo
import json

# --- MongoDB Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
AGENTS_DB_NAME = "reviews_db"
AGENTS_COLLECTION_NAME = "agent_configs"

def add_new_agent(agent_name: str, criteria: str, guidelines: str):
    """
    Adds a new agent configuration to the MongoDB database.

    Args:
        agent_name (str): The unique name for the new agent.
        criteria (str): The specific criteria for this agent to evaluate text against.
        guidelines (str): Official policy guidelines for this agent to follow.
    """
    client = None
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[AGENTS_DB_NAME]
        collection = db[AGENTS_COLLECTION_NAME]

        # Check if an agent with the same name already exists
        if collection.find_one({"agent_name": agent_name}):
            print(f"**❗ Agent '{agent_name}' already exists in the database. Skipping insertion.**")
            return

        agent_config = {
            "agent_name": agent_name,
            "criteria": criteria,
            "guidelines": guidelines
        }

        result = collection.insert_one(agent_config)
        print(f"**✅ Agent '{agent_name}' added successfully with ID:** {result.inserted_id}")

    except pymongo.errors.ConnectionFailure as e:
        print(f"**❌ MongoDB connection error:** {e}")
        print("Please ensure your MongoDB server is running on localhost:27017.")
    except Exception as e:
        print(f"**❌ An unexpected error occurred:** {e}")
    finally:
        if client:
            client.close()
            print("MongoDB connection closed.")

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Adding a New Agent ---")

    # Define the details for your new agent
    new_agent_name = "CulturalSensitivityReviewer"
    new_agent_criteria = """
    - Contains culturally insensitive language or stereotypes.
    - Misrepresents Pakistani cultural norms, traditions, or values.
    - Promotes negative or biased views against any ethnic or religious group within Pakistan.
    """
    new_agent_guidelines = """
    - Promote cultural harmony and respect.
    - Ensure accuracy in depicting diverse Pakistani cultures.
    - Avoid generalizations and reinforce positive aspects of cultural identity.
    """

    add_new_agent(new_agent_name, new_agent_criteria, new_agent_guidelines)

    # You can add more agents by calling the function again with different details
    # add_new_agent("HistoricalAccuracyReviewer", "Checks for historical inaccuracies...", "Refer to official historical records...")