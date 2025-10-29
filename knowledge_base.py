import json
import os
import pymongo
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.exceptions import OutputParserException
from config import MONGO_URI, KB_DB_NAME, KB_COLLECTION_NAME, CHROMA_DB_DIRECTORY
from llm_init import embeddings, llm1

# --- KNOWLEDGE BASE EXTRACTION AND VECTOR STORE INITIALIZATION ────────────────

def extract_knowledge_from_mongo(db_name: str, collection_name: str) -> List[Dict]:
    """
    Extracts knowledge entries from a MongoDB collection.

    Each entry is expected to have 'topic' and 'json_data' fields, where
    'json_data' is a JSON string containing official_narrative, key_points,
    sensitive_aspects, recommended_terminology, and authoritative_sources.

    Args:
        db_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        List[Dict]: A list of dictionaries, each representing a knowledge item
                    with parsed fields.
    """
    client = None
    extracted_knowledge = []
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[db_name]
        collection = db[collection_name]

        rows = collection.find({})

        for doc in rows:
            topic = doc.get("topic")
            json_data_str = doc.get("json_data")

            if topic and json_data_str:
                try:
                    data = json.loads(json_data_str)
                    if all(k in data for k in ["topic", "official_narrative", "key_points"]):
                        knowledge_item = {
                            "topic": data["topic"],
                            "official_narrative": data["official_narrative"],
                            "key_points": data["key_points"],
                            "sensitive_aspects": data.get("sensitive_aspects", []),
                            "recommended_terminology": data.get("recommended_terminology", {}),
                            "authoritative_sources": data.get("authoritative_sources", [])
                        }
                        extracted_knowledge.append(knowledge_item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for topic '{topic}': {e}")
                except Exception as e:
                    print(f"Error processing topic '{topic}': {e}")
            else:
                print(f"Warning: Skipping document with missing 'topic' or 'json_data': {doc}")
    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB connection error for KB: {e}")
        print("Please ensure MongoDB server is running on localhost:27017.")
    except Exception as e:
        print(f"An unexpected error occurred during KB extraction: {e}")
    finally:
        if client:
            client.close()
    return extracted_knowledge

knowledge_list = extract_knowledge_from_mongo(KB_DB_NAME, KB_COLLECTION_NAME)

if not os.path.exists(CHROMA_DB_DIRECTORY):
    print(f"Creating and persisting ChromaDB in '{CHROMA_DB_DIRECTORY}'...")
    if knowledge_list:
        docs = [
            Document(
                page_content=item["official_narrative"],
                metadata={
                    "topic": item["topic"],
                    "key_points": ", ".join(item["key_points"])
                }
            )
            for item in knowledge_list
        ]
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIRECTORY)
        print("ChromaDB created and persisted successfully.")
    else:
        print("No knowledge base data extracted from MongoDB. ChromaDB will not be created.")
        vectorstore = None # Ensure vectorstore is None if no data
else:
    print(f"Loading ChromaDB from '{CHROMA_DB_DIRECTORY}'...")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings)
    print("ChromaDB loaded successfully.")

# Define metadata field information for self-querying.
metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="The topic of the knowledge document (string)",
        type="string",
    ),
    AttributeInfo(
        name="key_points",
        description="Key points related to the topic (comma-separated string)",
        type="string",
    ),
]

document_content_description = "Knowledge Base official narratives and facts"

# Initialize the SelfQueryRetriever, enabling it to construct queries over the vector store's metadata
retriever = SelfQueryRetriever.from_llm(
    llm1,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
) if vectorstore else None # Only initialize if vectorstore exists

def get_relevant_info(query: str, k: int = 50) -> List[Dict]:
    """
    Retrieves relevant documents from the vector store based on a query
    and merges them with the full knowledge base data.
    """
    if not retriever:
        print("Retriever not initialized because ChromaDB was not created or loaded.")
        return []

    try:
        results = retriever.get_relevant_documents(query, k=k)
    except OutputParserException as e:
        print(f"Warning: SelfQueryRetriever failed with error: {e}. Falling back to similarity search.")
        results = vectorstore.similarity_search(query, k=k)
        
    unique_relevant_info = []
    seen_content = set()

    global knowledge_list
    if not knowledge_list:
        knowledge_list = extract_knowledge_from_mongo(KB_DB_NAME, KB_COLLECTION_NAME)

    if results:
        for doc in results:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                full_item = next((item for item in knowledge_list if item["official_narrative"] == content), None)

                if full_item:
                    unique_relevant_info.append({
                        "official_narrative": full_item["official_narrative"],
                        "topic": full_item["topic"],
                        "key_points": full_item["key_points"],
                        "sensitive_aspects": full_item["sensitive_aspects"],
                        "recommended_terminology": full_item["recommended_terminology"],
                        "authoritative_sources": full_item["authoritative_sources"]
                    })
        return unique_relevant_info
    else:
        return []