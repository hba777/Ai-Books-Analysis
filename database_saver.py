import json
import pymongo
import os
from datetime import datetime
from dotenv import load_dotenv
from bson.objectid import ObjectId
from typing import Dict, Any, Callable
from generate_prompt import build_prompt
from config import MONGO_URI, PDF_DB_NAME, PDF_COLLECTION_NAME
# اصلی LLM ماڈلز کو llm_init.py سے درآمد کریں
from llm_init import llm, eval_llm


load_dotenv()

# --- MongoDB Configuration ---
RESULTS_DB_NAME = os.getenv("RESULTS_DB_NAME1")
RESULTS_COLLECTION_NAME = os.getenv("RESULTS_COLLECTION_NAM1")

# -----------------------------
# 1️⃣ Clear Results Collection
# -----------------------------
def clear_results_collection():
    mongo_client = None
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        results_db = mongo_client[RESULTS_DB_NAME]
        results_collection = results_db[RESULTS_COLLECTION_NAME]

        delete_result = results_collection.delete_many({})
        print(f"🧹 Cleared {delete_result.deleted_count} documents from '{RESULTS_DB_NAME}.{RESULTS_COLLECTION_NAME}'.")

    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error during clearing: {e}")
    finally:
        if mongo_client:
            mongo_client.close()

# -----------------------------
# 2️⃣ Save Results to MongoDB
# -----------------------------
def save_results_to_mongo(
    chunk_uuid: str,
    doc_id: str,
    chunk_index: int,
    report_text: str,
    book_name: str,
    predicted_label: str,
    classification_scores: Dict[str, float],
    coordinates: Any,
    page_number: int,
    result_with_review: Dict,
    overall_chunk_status: str,
    agent_analysis_statuses: Dict
):
    mongo_client = None
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        results_db = mongo_client[RESULTS_DB_NAME]
        results_collection = results_db[RESULTS_COLLECTION_NAME]

        # Core document
        result_document = {
            "timestamp": datetime.now(),
            "book_id": doc_id,
            "Book Name": book_name,
            "Page Number": page_number,
            "Chunk_ID": chunk_uuid,
            "Chunk no.": chunk_index,
            "Text Analyzed": report_text,
            "coordinates": coordinates,
            "Predicted Label": predicted_label,
            "Predicted Label Confidence": classification_scores.get(predicted_label, 0.0),
            "overall_status": overall_chunk_status,
            "agent_responses": [] # New array to hold agent response documents
        }

        # Add individual agent data in the requested format
        main_node_output = result_with_review.get("main_node_output", {})
        for agent_name, agent_data in main_node_output.items():
            output_content = agent_data.get("output", {})
            
            # Create the exact document format you requested
            agent_response_doc = {
                "agent_name": agent_name,
                "response_content": output_content,
                "confidence": agent_data.get("confidence", 0),
                "retries": agent_data.get("retries", 0),
                "human_review": agent_data.get("human_review", False),
                "timestamp": datetime.now()
            }
            
            result_document["agent_responses"].append(agent_response_doc)

        results_collection.insert_one(result_document)
        print(f"✅ Merged results for chunk '{chunk_uuid}' saved to MongoDB in '{RESULTS_DB_NAME}.{RESULTS_COLLECTION_NAME}'.")

    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error while saving results: {e}")
    except Exception as e:
        print(f"❌ Unexpected error while saving results: {e}")
    finally:
        if mongo_client:
            mongo_client.close()

# -----------------------------
# 3️⃣ Update Chunk Analysis Status
# -----------------------------
def update_chunk_analysis_status(doc_id: str, chunk_id: str, analysis_status: str):
    mongo_client = None
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        p1_db = mongo_client[PDF_DB_NAME]
        chunks_collection = p1_db[PDF_COLLECTION_NAME]

        # Use ObjectId to query by MongoDB's unique _id field
        update_result = chunks_collection.update_one(
            {"doc_id": doc_id, "_id": ObjectId(chunk_id)},
            {"$set": {"analysis_status": analysis_status}}
        )

        if update_result.matched_count > 0:
            print(f"✅ Chunk '{chunk_id}' in document '{doc_id}' updated to '{analysis_status}'.")
        else:
            print(f"⚠️ Chunk '{chunk_id}' in document '{doc_id}' not found for status update.")

    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error during status update: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if mongo_client:
            mongo_client.close()

# -----------------------------
# 4️⃣ Dynamic Review Agent
# -----------------------------
def create_review_agent(
    agent_name: str,
    confidence_threshold: int,
    llm_model,
    eval_llm_model
) -> Callable[[Dict], Dict]:
    """
    Creates a specialized review agent that:
    1. Generates dynamic prompt using agent_name
    2. Invokes LLM model
    3. Evaluates confidence and decides human review
    """
    def agent_runner(state: Dict) -> Dict:
        report_text = state["report_text"]
        metadata = state.get("metadata", {})
        
        # Extract previous and next chunks from metadata for context
        previous_chunk = metadata.get("previous_chunk", "")
        next_chunk = metadata.get("next_chunk", "")

        retries = 0
        max_retries = 3
        human_review_flag = False
        parsed_output = {}

        while retries < max_retries:
            prompt = build_prompt(
                agent_name=agent_name,
                title=metadata.get("title", "N/A"),
                target_chunk=report_text,
                previous_chunk=previous_chunk,
                next_chunk=next_chunk
            )

            raw_response = llm_model.invoke(prompt).content
            try:
                parsed_output = json.loads(raw_response)
            except json.JSONDecodeError:
                human_review_flag = True
                break

            # Evaluate confidence
            eval_prompt = f"""
Evaluate the following prompt and response for correctness.
Prompt: {prompt}
Response: {raw_response}
Return JSON: {{"confidence": <score 0-100>}}
"""
            eval_response = eval_llm_model.invoke(eval_prompt).content
            try:
                eval_data = json.loads(eval_response)
                confidence = int(eval_data.get("confidence", 0))
            except Exception:
                confidence = 0

            if confidence >= confidence_threshold:
                break
            retries += 1
            if retries >= max_retries:
                human_review_flag = True

        return {
            "output": parsed_output,
            "confidence": confidence,
            "retries": retries,
            "human_review": human_review_flag
        }

    return agent_runner

# -----------------------------
# 5️⃣ Example Usage
# -----------------------------
if __name__ == "__main__":
    # Clear previous results
    clear_results_collection()

    # Example chunk data
    example_state = {
        "report_text": "Example text chunk from book...",
        "metadata": {"title": "Pakistan History 101"},
    }

    # اب اصلی LLM ماڈلز استعمال کریں
    # llm اور eval_llm پہلے ہی اوپر درآمد ہو چکے ہیں۔
    llm_model = llm
    eval_llm_model = eval_llm

    # Create agent
    agent = create_review_agent("HistoryPolicyAgent", 70, llm_model, eval_llm_model)
    result = agent(example_state)

    # Save to MongoDB
    save_results_to_mongo(
        chunk_uuid="chunk123",
        doc_id="doc001",
        chunk_index=1,
        report_text=example_state["report_text"],
        book_name=example_state["metadata"]["title"],
        predicted_label="Neutral",
        classification_scores={"Neutral": 90},
        coordinates=[],
        page_number=1,
        result_with_review={"main_node_output": {"HistoryPolicyAgent": result}},
        overall_chunk_status="Complete",
        agent_analysis_statuses={"HistoryPolicyAgent": "Done"}
    )

    # Update chunk status
    update_chunk_analysis_status("doc001", "chunk123", "Complete")